from pathlib import Path

import numpy as np
import torch
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from omegaconf import OmegaConf
from tqdm import tqdm


def get_diffusion_model(diffusion_root_path, ckpt, device):
    diffusion_root_path = Path(diffusion_root_path)
    diffusion_config_path = diffusion_root_path / "config.yaml"
    diffusion_ckpt_path = diffusion_root_path / ckpt
    print(f"using ckpt {ckpt}")

    config = OmegaConf.load(diffusion_config_path)
    diffusion_model = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))

    diffusion_ckpt = torch.load(diffusion_ckpt_path)

    diffusion_model.load_state_dict(diffusion_ckpt["diffusion_no_module"])
    diffusion_model.to(device)

    return diffusion_model, scheduler


@torch.no_grad()
def sampling_procedure(
    diffusion_model,
    scheduler,
    device,
    output_shape=(4, 56, 72, 56),
    num_of_samples=1,
):
    latent = torch.randn((num_of_samples,) + output_shape)
    latent = latent.to(device)

    model.eval()
    prompt_embeds = None
    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred = diffusion_model(
            x=latent,
            timesteps=torch.asarray((t,)).to(device),
            context=prompt_embeds,
        )
        latent, _ = scheduler.step(noise_pred, t, latent)

    return latent


root_path = Path("/home/sz9jt/data/generative_brain/diffusion/diffusion_tomhcp3dlatent_4ch")
ckpt = "checkpoint_no_module_350.pth"

exp_idx = 27
gpu_num = 4
device = torch.device(f"cuda:{exp_idx % gpu_num}")
print("using device", device)
print(f"experiment id: {exp_idx}")
model, scheduler = get_diffusion_model(root_path, ckpt, device)

for i in range(100):
    latent = sampling_procedure(model, scheduler, device)
    latent = latent.detach().cpu().numpy().squeeze()
    np.save(root_path / f"sample_ckpt350/sample_cuda{exp_idx}_{i}.npy", latent)
