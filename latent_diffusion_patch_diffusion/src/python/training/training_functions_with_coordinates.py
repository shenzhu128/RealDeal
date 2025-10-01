""" Training functions for the different models. """

from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from generative.losses.adversarial_loss import PatchAdversarialLoss
from pynvml.smi import nvidia_smi
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from util import log_ldm_sample_unconditioned, log_reconstructions


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")


# ----------------------------------------------------------------------------------------------------------------------
# Conditional Diffusion Model
# ----------------------------------------------------------------------------------------------------------------------
def train_cond_diffusion(
    model: nn.Module,
    # stage1: nn.Module,
    scheduler: nn.Module,
    # text_encoder,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    device: torch.device,
    run_dir: Path,
    task: str,
    # scale_factor: float = 1.0,
    gt_label="gt_image",
    cond_label="cond_image",
    coord_label="coordinates",
) -> float:
    scaler = GradScaler()

    val_loss = eval_cond_diffusion(
        model=model,
        # stage1=stage1,
        scheduler=scheduler,
        # text_encoder=text_encoder,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        task=task,
        # writer=writer_val,
        sample=False,
        # scale_factor=scale_factor,
        run_dir=run_dir,
        gt_label=gt_label,
        cond_label=cond_label,
        coord_label=coord_label,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_cond_diffusion(
            model=model,
            # stage1=stage1,
            scheduler=scheduler,
            # text_encoder=text_encoder,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            # writer=writer_train,
            scaler=scaler,
            # scale_factor=scale_factor,
            task=task,
            gt_label=gt_label,
            cond_label=cond_label,
            coord_label=coord_label,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_cond_diffusion(
                model=model,
                # stage1=stage1,
                scheduler=scheduler,
                # text_encoder=text_encoder,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                task=task,
                # writer=writer_val,
                sample=True,
                # sample=True if (epoch + 1) % (eval_freq * 2) == 0 else False,
                # scale_factor=scale_factor,
                run_dir=run_dir,
                epoch=epoch + 1,
                gt_label=gt_label,
                cond_label=cond_label,
                coord_label=coord_label,
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / f"checkpoint{epoch+1}.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(checkpoint, str(run_dir / f"bestmodel.pth"))

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "diffusion": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, str(run_dir / f"lastmodel.pth"))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_cond_diffusion(
    model: nn.Module,
    # stage1: nn.Module,
    scheduler: nn.Module,
    # text_encoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    # writer: SummaryWriter,
    scaler: GradScaler,
    task: str,
    # scale_factor: float = 1.0,
    gt_label="gt_image",
    cond_label="cond_image",
    coord_label="coordinates",
) -> None:
    model.train()

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x[gt_label].to(device)
        # reports = x["report"].to(device)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            # with torch.no_grad():
            #     e = stage1.encode_stage_2_inputs(images) * scale_factor

            # prompt_embeds = text_encoder(reports.squeeze(1))
            # prompt_embeds = prompt_embeds[0]
            coordinates = x[coord_label].to(device)

            noise = torch.randn_like(images).to(device)
            noisy_e = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
            if (task == "uncropping") and ("mask" in x.keys()):
                mask = x["mask"].to(device)
                noisy_e = noisy_e * mask + images * (1.0 - mask)
            cond_images = x[cond_label].to(device)
            concat_input = torch.cat([noisy_e, cond_images], dim=1).float()
            concat_input = concat_input.to(device)
            noise_pred = model(x=concat_input, timesteps=timesteps, coordinates=coordinates)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(images, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise

            if (task == "uncropping") and ("mask" in x.keys()):
                # Use the mask to compute the loss only on the masked region
                mask = x["mask"].to(device)
                noise_pred = noise_pred * mask
                target = target * mask
            loss = F.mse_loss(noise_pred.float(), target.float())

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        # writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)
        wandb.log({f"train/lr": get_lr(optimizer)})

        for k, v in losses.items():
            # writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)
            wandb.log({f"train/{k}": v.item()})

        pbar.set_postfix({"epoch": epoch, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})


@torch.no_grad()
def eval_cond_diffusion(
    model: nn.Module,
    # stage1: nn.Module,
    scheduler: nn.Module,
    # text_encoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    task: str,
    # writer: SummaryWriter,
    sample: bool = False,
    # scale_factor: float = 1.0,
    run_dir: Path = None,
    epoch: int = 0,
    gt_label="gt_image",
    cond_label="cond_image",
    coord_label="coordinates",
) -> float:
    model.eval()
    total_losses = OrderedDict()

    for x in loader:
        images = x[gt_label].to(device)
        # reports = x["report"].to(device)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        with autocast(enabled=True):
            # e = stage1.encode_stage_2_inputs(images) * scale_factor
            coordinates = x[coord_label].to(device)

            noise = torch.randn_like(images).to(device)
            noisy_e = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
            if (task == "uncropping") and ("mask" in x.keys()):
                mask = x["mask"].to(device)
                noisy_e = noisy_e * mask + images * (1.0 - mask)
            cond_images = x[cond_label].to(device)

            assert len(cond_images.shape) == 5
            concat_input = torch.cat([noisy_e, cond_images], dim=1).float()
            concat_input = concat_input.to(device)
            noise_pred = model(x=concat_input, timesteps=timesteps, coordinates=coordinates)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(images, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise

            if (task == "uncropping") and ("mask" in x.keys()):
                # Use the mask to compute the loss only on the masked region
                mask = x["mask"].to(device)
                noise_pred = noise_pred * mask
                target = target * mask
            loss = F.mse_loss(noise_pred.float(), target.float())

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        # writer.add_scalar(f"{k}", v, step)
        wandb.log({f"val/{k}": v})

    # if sample:
    #     if hasattr(loader.dataset, "fixed_scale"):
    #         scale_factor = loader.dataset.fixed_scale
    #     else:
    #         scale_factor = None

    #     log_ldm_sample_unconditioned(
    #         model=model,
    #         # stage1=stage1,
    #         scheduler=scheduler,
    #         # text_encoder=text_encoder,
    #         spatial_shape=tuple(images.shape[1:]),
    #         # writer=writer,
    #         step=step,
    #         device=device,
    #         scale_factor=scale_factor,
    #         run_dir=run_dir,
    #         epoch=epoch,
    #     )

    return total_losses["loss"]
