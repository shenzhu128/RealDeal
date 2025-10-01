import sys
from pathlib import Path

sys.path.append("..")

import argparse
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from datasets import combined_brain_1mm, combined_brain_17mm
from generative.metrics import MultiScaleSSIMMetric
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from models.autoencoderkl import AutoencoderKLDownsampleControl
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_diffusion_model(diffusion_root_path, ckpt, device, gpus=4):
    diffusion_root_path = Path(diffusion_root_path)
    diffusion_config_path = diffusion_root_path / "config.yaml"
    diffusion_ckpt_path = diffusion_root_path / f"checkpoint{ckpt}.pth"

    config = OmegaConf.load(diffusion_config_path)
    diffusion_model = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))

    device = device % gpus
    device = torch.device(f"cuda:{device}")
    diffusion_ckpt = torch.load(diffusion_ckpt_path)

    diffusion_model = torch.nn.DataParallel(diffusion_model)
    diffusion_model.load_state_dict(diffusion_ckpt["diffusion"])
    diffusion_model = diffusion_model.module
    diffusion_model.to(device)

    output_shape = config.dataset.params.output_shape
    ch = config.ldm.params.out_channels
    # ch = 3
    output_shape = (ch,) + tuple(output_shape)
    # test if config.dataset.params.fixed_scale exists
    if hasattr(config.dataset.params, "fixed_scale"):
        scale_factor = config.dataset.params.fixed_scale
    elif hasattr(config.dataset.params, "channel_scales"):
        scale_factor = config.dataset.params.channel_scales
        scale_factor = scale_factor[0]
    else:
        scale_factor = "None"

    return diffusion_model, scheduler, output_shape, scale_factor, device


@torch.no_grad()
def sampling_procedure(
    outpath,
    diffusion_model,
    scheduler,
    device,
    output_shape,
    scale_factor,
    idx,
    samples_per_gpu,
    device_rank,
    epoch,
):
    latent = torch.randn((1,) + output_shape)
    latent = latent.to(device)

    # idx = samples_per_gpu * device_rank + idx
    print(f"Sampling {idx}")

    if os.path.exists(outpath / f"sample_epoch{epoch}_fixscale{scale_factor}_{idx}.npy"):
        return

    prompt_embeds = None
    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred = diffusion_model(x=latent, timesteps=torch.asarray((t,)).to(device), context=prompt_embeds)
        latent, _ = scheduler.step(noise_pred, t, latent)

    #     nrow = len(latent)
    #     ncol = len(latent[0])
    #     x_slice = 24

    #     fig = plt.figure()
    #     for j in range(nrow * ncol):
    #         ax = fig.add_subplot(nrow, ncol, j + 1)
    #         ax.axis("off")
    #         ax.imshow(latent[j // ncol, j % ncol, x_slice].cpu().numpy(), cmap="gray")
    #     plt.savefig(str(outpath / f"sample_epoch{epoch}_{i}.png"), bbox_inches="tight")
    #     plt.close()

    np.save(str(outpath / f"sample_epoch{epoch}_fixscale{scale_factor}_{idx}.npy"), latent.cpu().numpy())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--diffusion_root_path", type=str, required=True)
    args.add_argument("--ckpt", type=str, required=True)
    args.add_argument("--device_rank", type=int, required=True)
    args.add_argument("--samples_per_gpu", type=int, required=True)
    args.add_argument("--need_to_sample", type=int, nargs="*", default=None)  # Accepts zero or more integers
    args = args.parse_args()

    diffusion_root_path = Path(args.diffusion_root_path)
    diffusion_model, scheduler, output_shape, scale_factor, device = get_diffusion_model(
        diffusion_root_path, args.ckpt, args.device_rank
    )
    print(
        f"root path: {diffusion_root_path} \n device {device} \n output shape {output_shape} \n scale factor {scale_factor}"
    )
    outpath = diffusion_root_path / f"sample-ckpt{args.ckpt}"
    os.makedirs(outpath, exist_ok=True)

    number_gpus = torch.cuda.device_count()
    sampled_items = os.listdir(outpath)
    need_to_sample = []
    print(f"Found {len(sampled_items)} samples in {outpath}")
    for i in range(number_gpus * args.samples_per_gpu):
        tmp = f"sample_epoch{args.ckpt}_fixscale{scale_factor}_{i}.npy"
        if tmp not in sampled_items:
            need_to_sample.append(i)
    print(f"Need to sample {len(need_to_sample)} items")
    print(need_to_sample)

    # partition the need_to_sample list based on device_rank
    # need_to_sample = need_to_sample[args.device_rank :: number_gpus]
    # need_to_sample = [4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    need_to_sample = args.need_to_sample
    print(f"Device {args.device_rank} will sample {len(need_to_sample)} items")
    print(f"Sampling items: {need_to_sample}")

    for i in need_to_sample:
        sampling_procedure(
            outpath,
            diffusion_model,
            scheduler,
            device,
            output_shape,
            scale_factor,
            i,
            args.samples_per_gpu,
            args.device_rank,
            args.ckpt,
        )
