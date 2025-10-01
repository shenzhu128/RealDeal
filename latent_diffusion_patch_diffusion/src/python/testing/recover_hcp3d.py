import sys

sys.path.append("../")
sys.path.append("../training")

import argparse
import importlib
import sys
from pathlib import Path

import datasets.hcp3d_uncropping_ds
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import recover_hcp3d_utils
import torch
from models import diffusion_model_unet
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    ds = datasets.hcp3d_uncropping_ds.TomsHCP3DTestPatchUncroppingWithoutCoordinatesDataset(
        "/home/sz9jt/manifold/sz9jt/HCP_preprocess"
    )

    device = torch.device(f"cuda:{str(args.gpu_idx)}" if torch.cuda.is_available() else "cpu")
    root_path = Path(args.root_path)
    checkpoint_name = args.checkpoint_name

    diffusion_model, scheduler, checkpoint_name = recover_hcp3d_utils.get_cond_diffusion_model(
        root_path, checkpoint_name
    )
    diffusion_model.to(device)

    config = OmegaConf.load(root_path / "config.yaml")
    patch_size = (64, 64, 64)

    torch.manual_seed(config.args.seed)

    # x     0 left      x right
    # y     0 posterior y anterior
    # z     0 inferior  z superior
    data = ds[0]
    final_output = np.zeros((224, 320, 256))

    recover_hcp3d_utils.recover_first_patch(
        data,
        scheduler,
        diffusion_model,
        final_output,
        patch_size,
        device,
    )

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

    tmp = checkpoint_name.replace(".pth", "")
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, root_path / f"{tmp}_output.nii.gz")

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

    tmp = checkpoint_name.replace(".pth", "")
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, root_path / f"{tmp}_output.nii.gz")

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

    tmp = checkpoint_name.replace(".pth", "")
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, root_path / f"{tmp}_output.nii.gz")

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

    tmp = checkpoint_name.replace(".pth", "")
    nibout = nib.Nifti1Image(final_output, np.eye(4))
    nib.save(nibout, root_path / f"{tmp}_output.nii.gz")
    gt = data["complete_gt_image"]
    nibout = nib.Nifti1Image(gt, np.eye(4))
    nib.save(nibout, root_path / f"{tmp}_gt.nii.gz")
    cond = data["complete_cond_image"]
    nibout = nib.Nifti1Image(cond, np.eye(4))
    nib.save(nibout, root_path / f"{tmp}_cond.nii.gz")
