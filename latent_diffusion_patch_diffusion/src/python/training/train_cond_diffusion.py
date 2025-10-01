""" Training script for the diffusion model in the latent space of the pretraine AEKL model. """

import argparse
import sys
import warnings
from pathlib import Path

sys.path.append("..")

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
    if torch.cuda.device_count() > 1:
        # stage1 = torch.nn.DataParallel(stage1)
        diffusion = torch.nn.DataParallel(diffusion)

    # print(f"Loading Stage 1 from checkpoint {args.stage1_uri}")
    # stage1_data = torch.load(args.stage1_uri)
    # stage1.load_state_dict(stage1_data["state_dict"])
    # stage1.eval()

    # stage1 = stage1.to(device)
    diffusion = diffusion.to(device)

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

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
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
