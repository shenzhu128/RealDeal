import sys

sys.path.append("../")
sys.path.append("../training")

import argparse
import importlib
import sys
import time
from pathlib import Path

import datasets.hcp3d_uncropping_ds
import einops
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import recover_hcp3d_utils
import torch
from models import diffusion_model_unet
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


class GroupNormRunningStats3d(torch.nn.InstanceNorm3d):
    __constants__ = torch.nn.InstanceNorm3d.__constants__ + [
        "num_groups",
        "num_channels",
    ]
    num_groups: int
    num_channels: int

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        self.num_groups = num_groups
        self.num_channels = num_channels
        if self.num_channels % self.num_groups != 0:
            raise ValueError(
                f"'num_channels' {self.num_channels} must be divisible by 'num_groups'" f" {self.num_groups}."
            )
        self.num_channels_per_group = num_channels // self.num_groups
        super().__init__(
            num_features=self.num_groups,
            eps=eps,
            affine=False,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        # If per-channel affines are requested, then instantiate them here. Otherwise,
        # just leave them as None as created in the _NormBase class.
        self.affine = affine
        if self.affine:
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.reset_parameters()

    def extra_repr(self) -> str:
        return ", ".join(
            [
                "{num_groups}",
                "{num_channels}",
                "eps={eps}",
                "affine={affine}",
                "track_running_stats={track_running_stats}",
            ]
        ).format(**self.__dict__)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        feature_dim = input.dim() - self._get_no_batch_dim()
        # Feature size should be num_groups, but the affine size should be num_channels.
        if (input.size(feature_dim) != (self.num_groups * self.num_channels_per_group)) and self.affine:
            raise ValueError(
                f"expected input's size at dim={feature_dim} to match"
                f" ({self.num_groups * self.num_channels_per_group}),"
                f" but got: {input.size(feature_dim)}."
            )

        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        return self._apply_instance_norm(input)

    def _apply_instance_norm(self, input):
        # Input order is very particular, if the goal is to stay consistent with the
        # existing instance and group norm layers in pytorch. This order has been
        # validated against instance norm
        # (num_groups=N, num_channels=N, affine=True, track_running_stats=True)
        # and group norm
        # (num_groups=N, num_channels=M, affine=True, track_running_stats=False) layers
        # in pytorch.
        # Can also check another github user who recreated the group norm layer in a
        # more readable way (compared to pytorch):
        # <https://github.com/RoyHEyono/Pytorch-GroupNorm/blob/db7c29bf506ab768a11de620a101af0615405cc7/groupnorm.py>
        x = einops.rearrange(
            input,
            "b (g c_g) x y z -> b g c_g x y z",
            g=self.num_groups,
            c_g=self.num_channels_per_group,
        )
        y = torch.nn.functional.instance_norm(
            x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=None,
            bias=None,
            use_input_stats=self.training or not self.track_running_stats,
            momentum=self.momentum if self.momentum is not None else 0.0,
            eps=self.eps,
        )
        y = einops.rearrange(
            y,
            "b g c_g x y z -> b (g c_g) x y z",
            g=self.num_groups,
            c_g=self.num_channels_per_group,
        )

        # Apply affine transform here instead of within the instance_norm function.
        if self.affine:
            y = y * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        type=str,
        default="/home/sz9jt/manifold/sz9jt/generative_brain/cond_diffusion/diffusion_tomshcp3d_maskedcond_finetune/",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="checkpoint1900.pth",
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--modified_group_norm_ckpt",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--data_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    ds = datasets.hcp3d_uncropping_ds.TomsHCP3DTestPatchUncroppingWithoutCoordinatesDataset(
        "/home/sz9jt/manifold/sz9jt/HCP_preprocess"
    )
    print("Dataset loaded", flush=True)

    device = torch.device(f"cuda:{str(args.gpu_idx)}" if torch.cuda.is_available() else "cpu")
    root_path = Path(args.root_path)
    checkpoint_name = args.checkpoint_name

    diffusion_model, scheduler, checkpoint_name = recover_hcp3d_utils.get_cond_diffusion_model(
        root_path, checkpoint_name
    )
    diffusion_model.to(device)

    config = OmegaConf.load(root_path / "config.yaml")
    patch_size = (64, 64, 64)

    torch.manual_seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    print("Using seed", args.sample_seed, flush=True)

    # x     0 left      x right
    # y     0 posterior y anterior
    # z     0 inferior  z superior
    data = ds[args.data_idx]
    print("Using data idx", args.data_idx, flush=True)
    final_output = np.zeros((224, 320, 256))

    def replace_groupnorm_with_custom_same_groups(module):
        for name, child in module.named_children():
            # print(child.__class__.__name__)
            if isinstance(child, torch.nn.GroupNorm):
                print(f"Replacing {name} (GroupNorm) with GroupNormRunningStats3d")

                new_gn = GroupNormRunningStats3d(
                    num_groups=child.num_groups,  # keep same groups
                    num_channels=child.num_channels,  # keep same channels
                    eps=child.eps,
                    affine=child.affine,
                    track_running_stats=True,  # or False, depending on your preference
                )

                # Copy affine parameters if affine=True
                if child.affine:
                    new_gn.weight.data = child.weight.data.clone()
                    new_gn.bias.data = child.bias.data.clone()

                setattr(module, name, new_gn)

                for name, param in new_gn.named_parameters():
                    print(f"Param → {name}: {param.shape}, device = {param.device}")

                for name, buffer in new_gn.named_buffers():
                    print(f"Buffer → {name}: {buffer.shape}, device = {buffer.device}")
            else:
                replace_groupnorm_with_custom_same_groups(child)

    replace_groupnorm_with_custom_same_groups(diffusion_model)
    output_path = Path(args.ckpt_path)
    ckpt_path = output_path / f"checkpoint{str(args.modified_group_norm_ckpt)}.pth"
    ckpt_data = torch.load(ckpt_path, map_location=device)
    diffusion_model.load_state_dict(ckpt_data["diffusion"])
    diffusion_model.to(device)
    # TODO Cannot use evan(), somehow the behavior of the group norm is different
    # after using eval(), need to check it
    # diffusion_model.eval()
    print("Model loaded", flush=True)
    print(f"Checkpoint loaded from {ckpt_path}", flush=True)

    stime = time.time()

    recover_hcp3d_utils.recover_first_patch(
        data,
        scheduler,
        diffusion_model,
        final_output,
        patch_size,
        device,
    )
    print("First patch recovered")
    tmp = f"ckpt{str(args.modified_group_norm_ckpt)}_sample{str(args.data_idx)}"
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, output_path / f"{tmp}_output.nii.gz")
    print("Saved first patch to:", flush=True)
    print(output_path / f"{tmp}_output.nii.gz")

    # go posterior
    recover_hcp3d_utils.recover_first_column_patches(
        data,
        scheduler,
        diffusion_model,
        final_output,
        patch_size,
        device,
        gt_label="go_posterior_gt_image",
        cond_label="go_posterior_cond_image",
        mask_label="go_posterior_mask",
        coord_label="go_posterior_coordinates",
    )
    print("Posterior patches recovered")
    tmp = f"ckpt{str(args.modified_group_norm_ckpt)}_sample{str(args.data_idx)}"
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, output_path / f"{tmp}_output.nii.gz")
    print("Saved posterior patches to:", flush=True)
    print(output_path / f"{tmp}_output.nii.gz")

    # go anterior
    recover_hcp3d_utils.recover_first_column_patches(
        data,
        scheduler,
        diffusion_model,
        final_output,
        patch_size,
        device,
        gt_label="go_anterior_gt_image",
        cond_label="go_anterior_cond_image",
        mask_label="go_anterior_mask",
        coord_label="go_anterior_coordinates",
    )
    print("Anterior patches recovered")
    tmp = f"ckpt{str(args.modified_group_norm_ckpt)}_sample{str(args.data_idx)}"
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, output_path / f"{tmp}_output.nii.gz")
    print("Saved anterior patches to:", flush=True)
    print(output_path / f"{tmp}_output.nii.gz")

    # go inferior
    recover_hcp3d_utils.recover_other_column_patches(
        data,
        scheduler,
        diffusion_model,
        final_output,
        patch_size,
        device,
        gt_label="go_inferior_gt_image",
        cond_label="go_inferior_cond_image",
        mask_label="go_inferior_mask",
        coord_label="go_inferior_coordinates",
    )
    print("Inferior patches recovered")
    tmp = f"ckpt{str(args.modified_group_norm_ckpt)}_sample{str(args.data_idx)}"
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, output_path / f"{tmp}_output.nii.gz")
    print("Saved inferior patches to:", flush=True)
    print(output_path / f"{tmp}_output.nii.gz")

    # go superior
    recover_hcp3d_utils.recover_other_column_patches(
        data,
        scheduler,
        diffusion_model,
        final_output,
        patch_size,
        device,
        gt_label="go_superior_gt_image",
        cond_label="go_superior_cond_image",
        mask_label="go_superior_mask",
        coord_label="go_superior_coordinates",
    )
    print("Superior patches recovered")
    tmp = f"ckpt{str(args.modified_group_norm_ckpt)}_sample{str(args.data_idx)}"
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, output_path / f"{tmp}_output.nii.gz")
    print("Saved superior patches to:", flush=True)
    print(output_path / f"{tmp}_output.nii.gz")

    # go left
    recover_hcp3d_utils.recover_slab_patches(
        data,
        scheduler,
        diffusion_model,
        final_output,
        patch_size,
        device,
        gt_label="go_left_gt_image",
        cond_label="go_left_cond_image",
        mask_label="go_left_mask",
        coord_label="go_left_coordinates",
    )
    print("Left patches recovered")
    tmp = f"ckpt{str(args.modified_group_norm_ckpt)}_sample{str(args.data_idx)}"
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, output_path / f"{tmp}_output.nii.gz")
    print("Saved left patches to:", flush=True)
    print(output_path / f"{tmp}_output.nii.gz")

    # go right
    recover_hcp3d_utils.recover_slab_patches(
        data,
        scheduler,
        diffusion_model,
        final_output,
        patch_size,
        device,
        gt_label="go_right_gt_image",
        cond_label="go_right_cond_image",
        mask_label="go_right_mask",
        coord_label="go_right_coordinates",
    )
    print("Right patches recovered")
    tmp = f"ckpt{str(args.modified_group_norm_ckpt)}_sample{str(args.data_idx)}"
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, output_path / f"{tmp}_output.nii.gz")
    print("Saved right patches to:", flush=True)
    print(output_path / f"{tmp}_output.nii.gz")

    gt = data["complete_gt_image"]
    nibout = nib.Nifti1Image(gt, np.eye(4))
    nib.save(nibout, output_path / f"{tmp}_gt.nii.gz")
    cond = data["complete_cond_image"]
    nibout = nib.Nifti1Image(cond, np.eye(4))
    nib.save(nibout, output_path / f"{tmp}_cond.nii.gz")

    etime = time.time()
    print("Time taken to recover:", etime - stime, flush=True)
    print("Done", flush=True)
