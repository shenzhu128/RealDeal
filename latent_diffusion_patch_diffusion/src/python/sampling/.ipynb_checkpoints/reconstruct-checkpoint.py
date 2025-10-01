from pathlib import Path

import sys

sys.path.append("..")

import pandas as pd
import torch
from generative.metrics import MultiScaleSSIMMetric
from generative.networks.nets import AutoencoderKL
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import argparse

from models.autoencoderkl import AutoencoderKLDownsampleControl

from datasets import combined_brain_1mm, combined_brain_17mm, lidc
from torch.utils.data import DataLoader


def run(args):
    config = OmegaConf.load(args.config_path)
    
    if args.dataset_type == "brain":
        test_ds = combined_brain_1mm.CombinedBrainDataset(
            root_dir=config.args.root_dir,
            type="test",
            return_name=True,
        )
    elif args.dataset_type == "LIDC":
        test_ds = lidc.LIDCDataset(root_dir=config.args.root_dir, 
                                   type="test",
                                   return_name=True)
        
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )
    
    set_determinism(seed=config.args.seed)

    print("Creating model...")
    device = torch.device("cuda")
    stage1 = AutoencoderKLDownsampleControl(**config["stage1"]["params"])
    stage1 = torch.nn.DataParallel(stage1)

    stage1.load_state_dict(torch.load(args.weight)["state_dict"])
    stage1 = stage1.to(device)
    stage1.eval()

    for batch in tqdm(test_loader):
        x = batch["data"].to(device)
        name = batch["name"]
        assert len(name) == 1
        name = name[0]

        if os.path.exists(os.path.join(args.outpath, name+".nii.gz")):
            continue

        with torch.no_grad():
            x_recon = stage1.module.reconstruct(x)

        outimg = x_recon.cpu().numpy().squeeze()
        assert outimg.shape == (128, 128, 128)
        outimg = nib.Nifti1Image(outimg, np.eye(4))
        nib.save(outimg, os.path.join(args.outpath, name+".nii.gz"))


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_type", type=str, required=True)
    args.add_argument("--config_path", type=str, required=True)
    args.add_argument("--weight", type=str, required=True)
    args.add_argument("--outpath", type=str, required=True)
    args.add_argument("--corrected", type=str, required=True)
    args = args.parse_args()
    
    run(args)