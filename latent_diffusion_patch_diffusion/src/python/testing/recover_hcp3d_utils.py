import sys
from pathlib import Path

sys.path.append("..")

import multiprocessing as mp
import os

import datasets
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch

# from datasets import combined_brain_1mm, combined_brain_17mm
from generative.metrics import MultiScaleSSIMMetric
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from models.autoencoderkl import AutoencoderKLDownsampleControl
from models.diffusion_model_unet import ModifiedDiffusionModelUNetWithCoordinateEmbedding
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_cond_diffusion_model(diffusion_root_path, ckpt):
    diffusion_root_path = Path(diffusion_root_path)
    diffusion_config_path = diffusion_root_path / "config.yaml"
    diffusion_ckpt_path = diffusion_root_path / str(ckpt)
    print(f"Diffusion path {diffusion_ckpt_path}")

    config = OmegaConf.load(diffusion_config_path)
    diffusion_model = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    diffusion_ckpt = torch.load(diffusion_ckpt_path)

    diffusion_model = torch.nn.DataParallel(diffusion_model)
    # diffusion_model.load_state_dict(diffusion_ckpt["diffusion"])
    diffusion_model.load_state_dict(diffusion_ckpt["diffusion"])

    checkpoint_name = str(diffusion_ckpt_path).split("/")[-1]

    return diffusion_model, scheduler, checkpoint_name


def get_cond_diffusion_model_with_coordinates(diffusion_root_path, ckpt):
    diffusion_root_path = Path(diffusion_root_path)
    diffusion_config_path = diffusion_root_path / "config.yaml"
    diffusion_ckpt_path = diffusion_root_path / str(ckpt)
    print(f"Diffusion path {diffusion_ckpt_path}")

    config = OmegaConf.load(diffusion_config_path)
    diffusion_model = ModifiedDiffusionModelUNetWithCoordinateEmbedding(**config["ldm"].get("params", dict()))
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    diffusion_ckpt = torch.load(diffusion_ckpt_path)

    diffusion_model = torch.nn.DataParallel(diffusion_model)
    # diffusion_model.load_state_dict(diffusion_ckpt["diffusion"])
    diffusion_model.load_state_dict(diffusion_ckpt["diffusion"])

    checkpoint_name = str(diffusion_ckpt_path).split("/")[-1]

    return diffusion_model, scheduler, checkpoint_name


def recover_first_patch(data, scheduler, diffusion_model, final_output, patch_size, device):
    with torch.no_grad():
        initial_patch = data["initial_patch_gt_image"]
        initial_cond = data["initial_patch_cond_image"]
        initial_coordinates = data["initial_patch_coordinates"]

        y = initial_cond[1:].copy()
        y = torch.from_numpy(y).unsqueeze(0).float().to(device)
        cond = initial_cond.copy()
        cond = torch.from_numpy(cond).unsqueeze(0).float().to(device)
        print(y.shape, cond.shape)
        prompt_embeds = None
        for t in tqdm(scheduler.timesteps, ncols=70):
            tmp_input = torch.cat([y, cond], dim=1).float().to(device)
            noise_pred = diffusion_model(x=tmp_input, timesteps=torch.asarray((t,)).to(device), context=prompt_embeds)
            y, _ = scheduler.step(noise_pred, t, y)
        tmp = y.cpu().numpy().squeeze()
        final_output[
            initial_coordinates[0] : initial_coordinates[0] + patch_size[0],
            initial_coordinates[1] : initial_coordinates[1] + patch_size[1],
            initial_coordinates[2] : initial_coordinates[2] + patch_size[2],
        ] += tmp


def recover_first_patch_with_coordinates(data, scheduler, diffusion_model, final_output, patch_size, device):
    with torch.no_grad():
        initial_patch = data["initial_patch_gt_image"]
        initial_cond = data["initial_patch_cond_image"]
        initial_coordinates = data["initial_patch_coordinates"]
        initial_coordinates_normalized = data["initial_patch_coordinates_normalized"]

        y = initial_cond[1:].copy()  # (1, 1, 64, 64, 64)
        y = torch.from_numpy(y).unsqueeze(0).float().to(device)
        cond = initial_cond.copy()
        cond = torch.from_numpy(cond).unsqueeze(0).float().to(device)  # (1, 2, 64, 64, 64)
        # (1, 1, 3)
        coord_normalized = torch.from_numpy(initial_coordinates_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
        print(y.shape, cond.shape, coord_normalized.shape)

        for t in tqdm(scheduler.timesteps, ncols=70):
            tmp_input = torch.cat([y, cond], dim=1).float().to(device)
            noise_pred = diffusion_model(
                x=tmp_input, timesteps=torch.asarray((t,)).to(device), coordinates=coord_normalized
            )
            y, _ = scheduler.step(noise_pred, t, y)
        tmp = y.cpu().numpy().squeeze()
        final_output[
            initial_coordinates[0] : initial_coordinates[0] + patch_size[0],
            initial_coordinates[1] : initial_coordinates[1] + patch_size[1],
            initial_coordinates[2] : initial_coordinates[2] + patch_size[2],
        ] += tmp


def recover_first_column_patches(
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
):
    with torch.no_grad():
        gt_imgs = data[gt_label]
        cond_imgs = data[cond_label]
        mask_imgs = data[mask_label]
        coordinates = data[coord_label]
        num = len(gt_imgs)
        for i in range(num):
            mask = mask_imgs[i].copy()  # (64, 64, 64)
            y = cond_imgs[i][1].copy()  # (64, 64, 64)
            cond = cond_imgs[i].copy()  # (2, 64, 64, 64) [blurry_cond, masked_cond]
            # we need to replace part in masked_cond with previous output
            coord = coordinates[i]  # (3,)
            previous_output = final_output[
                coord[0] : coord[0] + patch_size[0],
                coord[1] : coord[1] + patch_size[1],
                coord[2] : coord[2] + patch_size[2],
            ].copy()

            y = y * mask + previous_output * (1.0 - mask)
            cond[1] = cond[1] * mask + previous_output * (1.0 - mask)

            y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, 64, 64, 64)
            cond = torch.from_numpy(cond).unsqueeze(0).float().to(device)  # (1, 2, 64, 64, 64)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, 64, 64, 64)
            previous_output = (
                torch.from_numpy(previous_output).unsqueeze(0).unsqueeze(0).float().to(device)
            )  # (1, 1, 64, 64, 64)

            print(y.shape, cond.shape)
            prompt_embeds = None
            for t in tqdm(scheduler.timesteps, ncols=70):
                tmp_input = torch.cat([y, cond], dim=1).float().to(device)
                noise_pred = diffusion_model(
                    x=tmp_input, timesteps=torch.asarray((t,)).to(device), context=prompt_embeds
                )
                y, _ = scheduler.step(noise_pred, t, y)
                if mask is not None:
                    y = previous_output * (1.0 - mask) + mask * y
            tmp = y.cpu().numpy().squeeze()
            tmp = tmp * mask.cpu().numpy().squeeze()
            final_output[
                coord[0] : coord[0] + patch_size[0],
                coord[1] : coord[1] + patch_size[1],
                coord[2] : coord[2] + patch_size[2],
            ] += tmp


def recover_first_column_patches_with_coordinates(
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
    normalized_coord_label="go_posterior_coordinates_normalized",
):
    with torch.no_grad():
        gt_imgs = data[gt_label]
        cond_imgs = data[cond_label]
        mask_imgs = data[mask_label]
        coordinates = data[coord_label]
        normalized_coordinates = data[normalized_coord_label]
        num = len(gt_imgs)
        for i in range(num):
            mask = mask_imgs[i].copy()  # (64, 64, 64)
            y = cond_imgs[i][1].copy()  # (64, 64, 64)
            cond = cond_imgs[i].copy()  # (2, 64, 64, 64) [blurry_cond, masked_cond]
            # we need to replace part in masked_cond with previous output
            coord = coordinates[i]  # (3,)
            previous_output = final_output[
                coord[0] : coord[0] + patch_size[0],
                coord[1] : coord[1] + patch_size[1],
                coord[2] : coord[2] + patch_size[2],
            ].copy()

            y = y * mask + previous_output * (1.0 - mask)
            cond[1] = cond[1] * mask + previous_output * (1.0 - mask)

            y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, 64, 64, 64)
            cond = torch.from_numpy(cond).unsqueeze(0).float().to(device)  # (1, 2, 64, 64, 64)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, 64, 64, 64)
            previous_output = (
                torch.from_numpy(previous_output).unsqueeze(0).unsqueeze(0).float().to(device)
            )  # (1, 1, 64, 64, 64)
            normalized_coord = normalized_coordinates[i].copy()  # (3,)
            normalized_coord = (
                torch.from_numpy(normalized_coord).unsqueeze(0).unsqueeze(0).float().to(device)
            )  # (1, 1, 3)
            print(y.shape, cond.shape, normalized_coord.shape)

            for t in tqdm(scheduler.timesteps, ncols=70):
                tmp_input = torch.cat([y, cond], dim=1).float().to(device)
                noise_pred = diffusion_model(
                    x=tmp_input, timesteps=torch.asarray((t,)).to(device), coordinates=normalized_coord
                )
                y, _ = scheduler.step(noise_pred, t, y)
                if mask is not None:
                    y = previous_output * (1.0 - mask) + mask * y
            tmp = y.cpu().numpy().squeeze()
            tmp = tmp * mask.cpu().numpy().squeeze()
            final_output[
                coord[0] : coord[0] + patch_size[0],
                coord[1] : coord[1] + patch_size[1],
                coord[2] : coord[2] + patch_size[2],
            ] += tmp


def recover_other_column_patches(
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
):
    with torch.no_grad():
        all_gt = data[gt_label]
        all_cond = data[cond_label]
        all_mask = data[mask_label]
        all_coordinates = data[coord_label]
        num = len(all_gt)
        for i in range(num):
            column_mask = all_mask[i].copy()  # (N, 64, 64, 64)
            column_y = all_cond[i][:, 1].copy()  # (N, 64, 64, 64)
            column_cond = all_cond[i].copy()  # (N, 2, 64, 64, 64) [blurry_cond, masked_cond]
            column_coord = all_coordinates[i].copy()  # (N, 3)
            column_previous_outputs = []
            # we need to replace part in masked_cond with previous output
            for j in range(len(column_coord)):
                coord = column_coord[j]  # (3,)
                previous_output = final_output[
                    coord[0] : coord[0] + patch_size[0],
                    coord[1] : coord[1] + patch_size[1],
                    coord[2] : coord[2] + patch_size[2],
                ].copy()
                column_previous_outputs.append(previous_output)
                column_y[j] = column_y[j] * column_mask[j] + previous_output * (1.0 - column_mask[j])
                column_cond[j][1] = column_cond[j][1] * column_mask[j] + previous_output * (1.0 - column_mask[j])

            column_y = torch.from_numpy(column_y).unsqueeze(1).float().to(device)  # (N, 1, 64, 64, 64)
            column_cond = torch.from_numpy(column_cond).float().to(device)  # (N, 2, 64, 64, 64)
            column_mask = torch.from_numpy(column_mask).unsqueeze(1).float().to(device)  # (N, 1, 64, 64, 64)
            column_previous_outputs = np.array(column_previous_outputs)  # (N, 64, 64, 64)
            column_previous_outputs = (
                torch.from_numpy(column_previous_outputs).unsqueeze(1).float().to(device)
            )  # (N, 1, 64, 64, 64)

            prompt_embeds = None
            for t in tqdm(scheduler.timesteps, ncols=70):
                tmp_input = torch.cat([column_y, column_cond], dim=1).float().to(device)
                timesteps = [t] * len(column_coord)
                timesteps = torch.asarray(timesteps).to(device)
                noise_pred = diffusion_model(x=tmp_input, timesteps=timesteps, context=prompt_embeds)
                column_y, _ = scheduler.step(noise_pred, t, column_y)
                if column_mask is not None:
                    column_y = column_previous_outputs * (1.0 - column_mask) + column_mask * column_y

            tmp = column_y.cpu().numpy().squeeze()
            tmp = tmp * column_mask.cpu().numpy().squeeze()
            for j in range(len(column_coord)):
                coord = column_coord[j]  # (3,)
                final_output[
                    coord[0] : coord[0] + patch_size[0],
                    coord[1] : coord[1] + patch_size[1],
                    coord[2] : coord[2] + patch_size[2],
                ] += tmp[j]


def recover_other_column_patches_with_coordinates(
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
    normalized_coord_label="go_inferior_coordinates_normalized",
):
    with torch.no_grad():
        all_gt = data[gt_label]
        all_cond = data[cond_label]
        all_mask = data[mask_label]
        all_coordinates = data[coord_label]
        all_normalized_coordinates = data[normalized_coord_label]
        num = len(all_gt)
        for i in range(num):
            column_mask = all_mask[i].copy()  # (N, 64, 64, 64)
            column_y = all_cond[i][:, 1].copy()  # (N, 64, 64, 64)
            column_cond = all_cond[i].copy()  # (N, 2, 64, 64, 64) [blurry_cond, masked_cond]
            column_coord = all_coordinates[i].copy()  # (N, 3)
            column_normalized_coord = all_normalized_coordinates[i].copy()  # (N, 3)
            column_previous_outputs = []
            # we need to replace part in masked_cond with previous output
            for j in range(len(column_coord)):
                coord = column_coord[j]  # (3,)
                previous_output = final_output[
                    coord[0] : coord[0] + patch_size[0],
                    coord[1] : coord[1] + patch_size[1],
                    coord[2] : coord[2] + patch_size[2],
                ].copy()
                column_previous_outputs.append(previous_output)
                column_y[j] = column_y[j] * column_mask[j] + previous_output * (1.0 - column_mask[j])
                column_cond[j][1] = column_cond[j][1] * column_mask[j] + previous_output * (1.0 - column_mask[j])

            column_y = torch.from_numpy(column_y).unsqueeze(1).float().to(device)  # (N, 1, 64, 64, 64)
            column_cond = torch.from_numpy(column_cond).float().to(device)  # (N, 2, 64, 64, 64)
            column_mask = torch.from_numpy(column_mask).unsqueeze(1).float().to(device)  # (N, 1, 64, 64, 64)
            column_previous_outputs = np.array(column_previous_outputs)  # (N, 64, 64, 64)
            column_previous_outputs = (
                torch.from_numpy(column_previous_outputs).unsqueeze(1).float().to(device)
            )  # (N, 1, 64, 64, 64)
            column_normalized_coord = (
                torch.from_numpy(column_normalized_coord).unsqueeze(1).float().to(device)
            )  # (N, 1, 3)

            for t in tqdm(scheduler.timesteps, ncols=70):
                tmp_input = torch.cat([column_y, column_cond], dim=1).float().to(device)
                timesteps = [t] * len(column_coord)
                timesteps = torch.asarray(timesteps).to(device)
                noise_pred = diffusion_model(x=tmp_input, timesteps=timesteps, coordinates=column_normalized_coord)
                column_y, _ = scheduler.step(noise_pred, t, column_y)
                if column_mask is not None:
                    column_y = column_previous_outputs * (1.0 - column_mask) + column_mask * column_y

            tmp = column_y.cpu().numpy().squeeze()
            tmp = tmp * column_mask.cpu().numpy().squeeze()
            for j in range(len(column_coord)):
                coord = column_coord[j]  # (3,)
                final_output[
                    coord[0] : coord[0] + patch_size[0],
                    coord[1] : coord[1] + patch_size[1],
                    coord[2] : coord[2] + patch_size[2],
                ] += tmp[j]


def recover_slab_patches(
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
):
    with torch.no_grad():
        all_gt = data[gt_label]  # (3/2, 20, 64, 64, 64)
        all_cond = data[cond_label]  # (3/2, 20, 2, 64, 64, 64)
        all_mask = data[mask_label]  # (3/2, 20, 64, 64, 64)
        all_coordinates = data[coord_label]  # (3/2, 20, 3)
        num = len(all_gt)
        for i in range(num):
            slab_mask = all_mask[i].copy()  # (N, 64, 64, 64)
            slab_y = all_cond[i][:, 1].copy()  # (N, 64, 64, 64)
            slab_cond = all_cond[i].copy()  # (N, 2, 64, 64, 64) [blurry_cond, masked_cond]
            slab_coord = all_coordinates[i].copy()  # (N, 3)
            slab_previous_outputs = []
            # we need to replace part in masked_cond with previous output
            for j in range(len(slab_coord)):
                coord = slab_coord[j]  # (3,)
                previous_output = final_output[
                    coord[0] : coord[0] + patch_size[0],
                    coord[1] : coord[1] + patch_size[1],
                    coord[2] : coord[2] + patch_size[2],
                ].copy()
                slab_previous_outputs.append(previous_output)
                slab_y[j] = slab_y[j] * slab_mask[j] + previous_output * (1.0 - slab_mask[j])
                slab_cond[j][1] = slab_cond[j][1] * slab_mask[j] + previous_output * (1.0 - slab_mask[j])

            slab_y = torch.from_numpy(slab_y).unsqueeze(1).float().to(device)  # (N, 1, 64, 64, 64)
            slab_cond = torch.from_numpy(slab_cond).float().to(device)  # (N, 2, 64, 64, 64)
            slab_mask = torch.from_numpy(slab_mask).unsqueeze(1).float().to(device)  # (N, 1, 64, 64, 64)
            slab_previous_outputs = np.array(slab_previous_outputs)  # (N, 64, 64, 64)
            slab_previous_outputs = (
                torch.from_numpy(slab_previous_outputs).unsqueeze(1).float().to(device)
            )  # (N, 1, 64, 64, 64)

            prompt_embeds = None
            for t in tqdm(scheduler.timesteps, ncols=70):
                tmp_input = torch.cat([slab_y, slab_cond], dim=1).float().to(device)
                timesteps = [t] * len(slab_coord)
                timesteps = torch.asarray(timesteps).to(device)
                noise_pred = diffusion_model(x=tmp_input, timesteps=timesteps, context=prompt_embeds)
                slab_y, _ = scheduler.step(noise_pred, t, slab_y)
                if slab_mask is not None:
                    slab_y = slab_previous_outputs * (1.0 - slab_mask) + slab_mask * slab_y

            tmp = slab_y.cpu().numpy().squeeze()
            tmp = tmp * slab_mask.cpu().numpy().squeeze()
            for j in range(len(slab_coord)):
                coord = slab_coord[j]  # (3,)
                final_output[
                    coord[0] : coord[0] + patch_size[0],
                    coord[1] : coord[1] + patch_size[1],
                    coord[2] : coord[2] + patch_size[2],
                ] += tmp[j]


def recover_slab_patches_with_coordinates(
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
    normalized_coord_label="go_left_coordinates_normalized",
):
    with torch.no_grad():
        all_gt = data[gt_label]  # (3/2, 20, 64, 64, 64)
        all_cond = data[cond_label]  # (3/2, 20, 2, 64, 64, 64)
        all_mask = data[mask_label]  # (3/2, 20, 64, 64, 64)
        all_coordinates = data[coord_label]  # (3/2, 20, 3)
        all_normalized_coordinates = data[normalized_coord_label]  # (3/2, 20, 3)
        num = len(all_gt)
        for i in range(num):
            slab_mask = all_mask[i].copy()  # (N, 64, 64, 64)
            slab_y = all_cond[i][:, 1].copy()  # (N, 64, 64, 64)
            slab_cond = all_cond[i].copy()  # (N, 2, 64, 64, 64) [blurry_cond, masked_cond]
            slab_coord = all_coordinates[i].copy()  # (N, 3)
            slab_normalized_coord = all_normalized_coordinates[i].copy()  # (N, 3)
            slab_previous_outputs = []
            # we need to replace part in masked_cond with previous output
            for j in range(len(slab_coord)):
                coord = slab_coord[j]  # (3,)
                previous_output = final_output[
                    coord[0] : coord[0] + patch_size[0],
                    coord[1] : coord[1] + patch_size[1],
                    coord[2] : coord[2] + patch_size[2],
                ].copy()
                slab_previous_outputs.append(previous_output)
                slab_y[j] = slab_y[j] * slab_mask[j] + previous_output * (1.0 - slab_mask[j])
                slab_cond[j][1] = slab_cond[j][1] * slab_mask[j] + previous_output * (1.0 - slab_mask[j])

            slab_y = torch.from_numpy(slab_y).unsqueeze(1).float().to(device)  # (N, 1, 64, 64, 64)
            slab_cond = torch.from_numpy(slab_cond).float().to(device)  # (N, 2, 64, 64, 64)
            slab_mask = torch.from_numpy(slab_mask).unsqueeze(1).float().to(device)  # (N, 1, 64, 64, 64)
            slab_previous_outputs = np.array(slab_previous_outputs)  # (N, 64, 64, 64)
            slab_previous_outputs = (
                torch.from_numpy(slab_previous_outputs).unsqueeze(1).float().to(device)
            )  # (N, 1, 64, 64, 64)
            slab_normalized_coord = torch.from_numpy(slab_normalized_coord).unsqueeze(1).float().to(device)
            # (N, 1, 3)

            for t in tqdm(scheduler.timesteps, ncols=70):
                tmp_input = torch.cat([slab_y, slab_cond], dim=1).float().to(device)
                timesteps = [t] * len(slab_coord)
                timesteps = torch.asarray(timesteps).to(device)
                noise_pred = diffusion_model(x=tmp_input, timesteps=timesteps, coordinates=slab_normalized_coord)
                slab_y, _ = scheduler.step(noise_pred, t, slab_y)
                if slab_mask is not None:
                    slab_y = slab_previous_outputs * (1.0 - slab_mask) + slab_mask * slab_y

            tmp = slab_y.cpu().numpy().squeeze()
            tmp = tmp * slab_mask.cpu().numpy().squeeze()
            for j in range(len(slab_coord)):
                coord = slab_coord[j]  # (3,)
                final_output[
                    coord[0] : coord[0] + patch_size[0],
                    coord[1] : coord[1] + patch_size[1],
                    coord[2] : coord[2] + patch_size[2],
                ] += tmp[j]
