""" Training functions for the different models. """
from collections import OrderedDict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
# AUTOENCODER KL
# ----------------------------------------------------------------------------------------------------------------------
def train_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    device: torch.device,
    run_dir: Path,
    adv_weight: float,
    perceptual_weight: float,
    kl_weight: float,
    adv_start: int,
) -> float:
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_aekl(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        kl_weight=kl_weight,
        adv_weight=adv_weight if start_epoch >= adv_start else 0.0,
        perceptual_weight=perceptual_weight,
        run_dir=run_dir,
        epoch=0,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    for epoch in range(start_epoch, n_epochs):
        train_epoch_aekl(
            model=model,
            discriminator=discriminator,
            perceptual_loss=perceptual_loss,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            kl_weight=kl_weight,
            adv_weight=adv_weight if epoch >= adv_start else 0.0,
            perceptual_weight=perceptual_weight,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
            run_dir=run_dir,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_aekl(
                model=model,
                discriminator=discriminator,
                perceptual_loss=perceptual_loss,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                kl_weight=kl_weight,
                adv_weight=adv_weight if epoch >= adv_start else 0.0,
                perceptual_weight=perceptual_weight,
                run_dir=run_dir,
                epoch=epoch + 1,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / f"checkpoint{epoch+1}.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss

        wandb.alert(title=f"Training epoch {epoch+1} finished", text=f"Val loss: {val_loss:.4f}")

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    # writer: SummaryWriter,
    kl_weight: float,
    adv_weight: float,
    perceptual_weight: float,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
    run_dir: Path,
) -> None:
    model.train()
    discriminator.train()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["data"].to(device)

        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # DISCRIMINATOR
        if adv_weight > 0:
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                d_loss = adv_weight * discriminator_loss
                d_loss = d_loss.mean()

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            discriminator_loss = torch.tensor([0.0]).to(device)

        losses["d_loss"] = discriminator_loss

        # writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
        # writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
        # for k, v in losses.items():
        #     writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)
        wandb.log({f"train/lr_g": get_lr(optimizer_g)})
        wandb.log({f"train/lr_d": get_lr(optimizer_d)})
        for k, v in losses.items():
            wandb.log({f"train/{k}": v.item()})

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "l1_loss": f"{losses['l1_loss'].item():.6f}",
                "p_loss": f"{losses['p_loss'].item():.6f}",
                "g_loss": f"{losses['g_loss'].item():.6f}",
                "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}",
                "lr_d": f"{get_lr(optimizer_d):.6f}",
            },
        )


@torch.no_grad()
def eval_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    # writer: SummaryWriter,
    kl_weight: float,
    adv_weight: float,
    perceptual_weight: float,
    run_dir: Path,
    epoch: int,
) -> float:
    model.eval()
    discriminator.eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    total_losses = OrderedDict()
    for x in loader:
        images = x["data"].to(device)

        with autocast(enabled=True):
            # GENERATOR
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # DISCRIMINATOR
            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            else:
                discriminator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()
            d_loss = discriminator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
                d_loss=d_loss,
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        wandb.log({f"val/{k}": v})
        # writer.add_scalar(f"{k}", v, step)

    # log_reconstructions(
    #     image=images,
    #     reconstruction=reconstruction,
    #     step=step,
    # )

    images = images.detach().cpu().numpy()
    reconstruction = reconstruction.detach().cpu().numpy()

    output_path = run_dir / "images"
    output_path.mkdir(exist_ok=True, parents=True)
    fig, axes = plt.subplots(2, len(images))
    for i in range(len(images)):
        axes[0, i].imshow(images[i, 0, :, :, 100], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstruction[i, 0, :, :, 100], cmap="gray")
        axes[1, i].axis("off")
    tmp_loss = total_losses["l1_loss"]
    plt.savefig(output_path / f"val_epoch{epoch}_step{step}_loss{tmp_loss}.png", bbox_inches="tight")

    return total_losses["l1_loss"]


# ----------------------------------------------------------------------------------------------------------------------
# VANILLA AUTOENCODER KL
# ----------------------------------------------------------------------------------------------------------------------
def train_vanilla_aekl(
    model: nn.Module,
    # discriminator: nn.Module,
    # perceptual_loss: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    # optimizer_d: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    device: torch.device,
    run_dir: Path,
    # adv_weight: float,
    # perceptual_weight: float,
    kl_weight: float,
    # adv_start: int,
) -> float:
    scaler_g = GradScaler()
    # scaler_d = GradScaler()

    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_vanilla_aekl(
        model=model,
        # discriminator=discriminator,
        # perceptual_loss=perceptual_loss,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        kl_weight=kl_weight,
        # adv_weight=adv_weight if start_epoch >= adv_start else 0.0,
        # perceptual_weight=perceptual_weight,
        run_dir=run_dir,
        epoch=0,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    for epoch in range(start_epoch, n_epochs):
        train_epoch_vanilla_aekl(
            model=model,
            # discriminator=discriminator,
            # perceptual_loss=perceptual_loss,
            loader=train_loader,
            optimizer_g=optimizer_g,
            # optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            kl_weight=kl_weight,
            # adv_weight=adv_weight if epoch >= adv_start else 0.0,
            # perceptual_weight=perceptual_weight,
            scaler_g=scaler_g,
            # scaler_d=scaler_d,
            run_dir=run_dir,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_vanilla_aekl(
                model=model,
                # discriminator=discriminator,
                # perceptual_loss=perceptual_loss,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                kl_weight=kl_weight,
                # adv_weight=adv_weight if epoch >= adv_start else 0.0,
                # perceptual_weight=perceptual_weight,
                run_dir=run_dir,
                epoch=epoch + 1,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                # "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                # "optimizer_d": optimizer_d.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / f"checkpoint{epoch+1}.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_vanilla_aekl(
    model: nn.Module,
    # discriminator: nn.Module,
    # perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    # optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    # writer: SummaryWriter,
    kl_weight: float,
    # adv_weight: float,
    # perceptual_weight: float,
    scaler_g: GradScaler,
    # scaler_d: GradScaler,
    run_dir: Path,
) -> None:
    model.train()
    # discriminator.train()

    # adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["data"].to(device)

        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            # p_loss = perceptual_loss(reconstruction.float(), images.float())

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            # if adv_weight > 0:
            #     logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            #     generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            # else:
            #     generator_loss = torch.tensor([0.0]).to(device)

            # loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss
            loss = l1_loss + kl_weight * kl_loss
            loss = loss.mean()
            l1_loss = l1_loss.mean()
            # p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            # g_loss = generator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                # p_loss=p_loss,
                kl_loss=kl_loss,
                # g_loss=g_loss,
            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # # DISCRIMINATOR
        # if adv_weight > 0:
        #     optimizer_d.zero_grad(set_to_none=True)

        #     with autocast(enabled=True):
        #         logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        #         loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        #         logits_real = discriminator(images.contiguous().detach())[-1]
        #         loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        #         discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        #         d_loss = adv_weight * discriminator_loss
        #         d_loss = d_loss.mean()

        #     scaler_d.scale(d_loss).backward()
        #     scaler_d.unscale_(optimizer_d)
        #     torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
        #     scaler_d.step(optimizer_d)
        #     scaler_d.update()
        # else:
        #     discriminator_loss = torch.tensor([0.0]).to(device)

        # losses["d_loss"] = discriminator_loss

        # writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
        # writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
        # for k, v in losses.items():
        #     writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)
        wandb.log({f"train/lr_g": get_lr(optimizer_g)})
        # wandb.log({f"train/lr_d": get_lr(optimizer_d)})
        for k, v in losses.items():
            wandb.log({f"train/{k}": v.item()})

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "l1_loss": f"{losses['l1_loss'].item():.6f}",
                # "p_loss": f"{losses['p_loss'].item():.6f}",
                # "g_loss": f"{losses['g_loss'].item():.6f}",
                # "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}",
                # "lr_d": f"{get_lr(optimizer_d):.6f}",
            },
        )


@torch.no_grad()
def eval_vanilla_aekl(
    model: nn.Module,
    # discriminator: nn.Module,
    # perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    # writer: SummaryWriter,
    kl_weight: float,
    # adv_weight: float,
    # perceptual_weight: float,
    run_dir: Path,
    epoch: int,
) -> float:
    model.eval()
    # discriminator.eval()

    # adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    total_losses = OrderedDict()
    for x in loader:
        images = x["data"].to(device)

        with autocast(enabled=True):
            # GENERATOR
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            # p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            # if adv_weight > 0:
            #     logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            #     generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            # else:
            #     generator_loss = torch.tensor([0.0]).to(device)

            # # DISCRIMINATOR
            # if adv_weight > 0:
            #     logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            #     loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            #     logits_real = discriminator(images.contiguous().detach())[-1]
            #     loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            #     discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            # else:
            #     discriminator_loss = torch.tensor([0.0]).to(device)

            # loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss
            loss = l1_loss + kl_weight * kl_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            # p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            # g_loss = generator_loss.mean()
            # d_loss = discriminator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                # p_loss=p_loss,
                kl_loss=kl_loss,
                # g_loss=g_loss,
                # d_loss=d_loss,
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        wandb.log({f"val/{k}": v})
        # writer.add_scalar(f"{k}", v, step)

    # log_reconstructions(
    #     image=images,
    #     reconstruction=reconstruction,
    #     step=step,
    # )

    images = images.detach().cpu().numpy()
    reconstruction = reconstruction.detach().cpu().numpy()

    output_path = run_dir / "images"
    output_path.mkdir(exist_ok=True, parents=True)
    fig, axes = plt.subplots(2, len(images) * 3)
    for i in range(len(images)):
        start = i * 3
        axes[0, start].imshow(np.rot90(images[i, 0, :, :, 100]), cmap="gray")
        axes[0, start].axis("off")
        axes[1, start].imshow(np.rot90(reconstruction[i, 0, :, :, 100]), cmap="gray")
        axes[1, start].axis("off")

        axes[0, start + 1].imshow(np.rot90(images[i, 0, :, 120, :]), cmap="gray")
        axes[0, start + 1].axis("off")
        axes[1, start + 1].imshow(np.rot90(reconstruction[i, 0, :, 120, :]), cmap="gray")
        axes[1, start + 1].axis("off")

        axes[0, start + 2].imshow(np.rot90(images[i, 0, 100, :, :]), cmap="gray")
        axes[0, start + 2].axis("off")
        axes[1, start + 2].imshow(np.rot90(reconstruction[i, 0, 100, :, :]), cmap="gray")
        axes[1, start + 2].axis("off")
    tmp_loss = total_losses["l1_loss"]
    plt.tight_layout()
    plt.savefig(output_path / f"val_epoch_{epoch}_step{step}_loss{tmp_loss}.png", bbox_inches="tight")

    return total_losses["l1_loss"]


# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion Model
# ----------------------------------------------------------------------------------------------------------------------
def train_ldm(
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
    # scale_factor: float = 1.0,
) -> float:
    scaler = GradScaler()

    val_loss = eval_ldm(
        model=model,
        # stage1=stage1,
        scheduler=scheduler,
        # text_encoder=text_encoder,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        # writer=writer_val,
        sample=False,
        # scale_factor=scale_factor,
        run_dir=run_dir,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_ldm(
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
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_ldm(
                model=model,
                # stage1=stage1,
                scheduler=scheduler,
                # text_encoder=text_encoder,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                # writer=writer_val,
                sample=True,
                # sample=True if (epoch + 1) % (eval_freq * 2) == 0 else False,
                # scale_factor=scale_factor,
                run_dir=run_dir,
                epoch=epoch + 1,
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
                torch.save(checkpoint, str(run_dir / "best_model.pth"))

        checkpoint = {
            "epoch": epoch + 1,
            "diffusion": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, str(run_dir / f"lastest_ckpt.pth"))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_ldm(
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
    # scale_factor: float = 1.0,
) -> None:
    model.train()

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["image"].to(device)
        # reports = x["report"].to(device)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            # with torch.no_grad():
            #     e = stage1.encode_stage_2_inputs(images) * scale_factor

            # prompt_embeds = text_encoder(reports.squeeze(1))
            # prompt_embeds = prompt_embeds[0]
            prompt_embeds = None

            noise = torch.randn_like(images).to(device)
            noisy_e = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
            noise_pred = model(x=noisy_e, timesteps=timesteps, context=prompt_embeds)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(images, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
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
def eval_ldm(
    model: nn.Module,
    # stage1: nn.Module,
    scheduler: nn.Module,
    # text_encoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    # writer: SummaryWriter,
    sample: bool = False,
    # scale_factor: float = 1.0,
    run_dir: Path = None,
    epoch: int = 0,
) -> float:
    model.eval()
    total_losses = OrderedDict()

    for x in loader:
        images = x["image"].to(device)
        # reports = x["report"].to(device)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        with autocast(enabled=True):
            # e = stage1.encode_stage_2_inputs(images) * scale_factor
            prompt_embeds = None

            noise = torch.randn_like(images).to(device)
            noisy_e = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
            noise_pred = model(x=noisy_e, timesteps=timesteps, context=prompt_embeds)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(images, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
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

    if sample:
        if hasattr(loader.dataset, "fixed_scale"):
            scale_factor = loader.dataset.fixed_scale
        else:
            scale_factor = None

        log_ldm_sample_unconditioned(
            model=model,
            # stage1=stage1,
            scheduler=scheduler,
            # text_encoder=text_encoder,
            spatial_shape=tuple(images.shape[1:]),
            # writer=writer,
            step=step,
            device=device,
            scale_factor=scale_factor,
            run_dir=run_dir,
            epoch=epoch,
        )

    return total_losses["loss"]
