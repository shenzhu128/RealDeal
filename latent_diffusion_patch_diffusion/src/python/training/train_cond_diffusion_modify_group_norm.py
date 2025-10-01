""" Training script for the diffusion model in the latent space of the pretraine AEKL model. """

import argparse
import sys
import warnings
from pathlib import Path

sys.path.append("..")

import einops
import torch
import torch.optim as optim
import training_functions_cond
import wandb
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

# from models.autoencoderkl import AutoencoderKLDownsampleControl
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from util import get_obj_from_str

warnings.filterwarnings("ignore")


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Location to save model checkpoints.")
    parser.add_argument("--run_dir", help="Location of the run.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--project", type=str, required=True, help="Wandb project name.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--enable_wandb", type=int, default=1, help="Enable wandb logging.")
    parser.add_argument("--task", type=str, default="uncropping", help="Task to train on.")
    # parser.add_argument("--stage1_uri", help="Path readable by load_model.")
    # parser.add_argument("--scale_factor", type=float, help="Path readable by load_model.")
    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    if not args.enable_wandb:
        wandb.init(mode="disabled")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / args.run_dir
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Get Models
    # print(f"Constructing Stage 1 from {args.stage1_config}")
    # stage1_config = OmegaConf.load(args.stage1_config)
    # stage1 = AutoencoderKLDownsampleControl(**stage1_config["stage1"]["params"])

    print("Creating model...")
    config = OmegaConf.load(args.config_file)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))

    # text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")

    # print(f"Loading Stage 1 from checkpoint {args.stage1_uri}")
    # stage1_data = torch.load(args.stage1_uri)
    # stage1.load_state_dict(stage1_data["state_dict"])
    # stage1.eval()
    # stage1 = stage1.to(device)
    # diffusion = diffusion.to(device)

    optimizer = optim.AdamW(diffusion.parameters(), lr=config["ldm"]["base_lr"])

    # Get Data
    train_ds = get_obj_from_str(config.dataset.target)(**config.dataset.params, type="train")
    val_ds = get_obj_from_str(config.dataset.target)(**config.dataset.params, type="train")
    # make subset of val_ds
    val_ds = torch.utils.data.Subset(val_ds, range(0, 2))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Replacing GroupNorm with GroupNormRunningStats3d
    print("Replacing GroupNorm with GroupNormRunningStats3d")

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
                # print device of each parameter
            else:
                replace_groupnorm_with_custom_same_groups(child)

    replace_groupnorm_with_custom_same_groups(diffusion)

    if torch.cuda.device_count() > 1:
        diffusion = torch.nn.DataParallel(diffusion)
    diffusion.to(device)

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "lastmodel.pth"))
        # see if checkpoint is using DataParallel
        if "module." in list(checkpoint["diffusion"].keys())[0] and (torch.cuda.device_count() == 1):
            # remove module. from keys
            print(f"Removing module. from keys")
            new_state_dict = {}
            for k, v in checkpoint["diffusion"].items():
                new_key = k.replace("module.", "")
                new_state_dict[new_key] = v
            checkpoint["diffusion"] = new_state_dict
        elif "module." not in list(checkpoint["diffusion"].keys())[0] and (torch.cuda.device_count() > 1):
            print(f"Adding module. to keys")
            new_state_dict = {}
            for k, v in checkpoint["diffusion"].items():
                new_key = "module." + k
                new_state_dict[new_key] = v
            checkpoint["diffusion"] = new_state_dict
        diffusion.load_state_dict(checkpoint["diffusion"])
        # Issue loading optimizer https://github.com/pytorch/pytorch/issues/2830
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print(f"No checkpoint found.")

    # running_mean, running_var -> not optimized; they are statistics
    # that are updated during training, so they're buffers (saved in state_dict but not optimized)
    # weight, bias -> optimized; they are parameters that are learned during training
    # they are optimized by the optimizer and saved in the state_dict
    diffusion.to(device)

    # Logging
    dict_config = OmegaConf.to_container(config, resolve=True)
    dict_config["args"] = vars(args)
    wandb.init(project=args.project, config=dict_config, name=args.run_dir)
    wandb.init(mode="disabled")
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(dict_config, f)
    print(f"Config saved to {str(run_dir / 'config.yaml')}")

    # Train model
    print(f"Starting Training")
    print(f"Using task {args.task}")
    val_loss = training_functions_cond.train_cond_diffusion(
        model=diffusion,
        # stage1=stage1,
        scheduler=scheduler,
        # text_encoder=text_encoder,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        device=device,
        run_dir=run_dir,
        task=args.task,
        # scale_factor=args.scale_factor,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
