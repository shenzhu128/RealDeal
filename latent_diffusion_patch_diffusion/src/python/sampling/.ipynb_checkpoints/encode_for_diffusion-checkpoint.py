import sys

sys.path.append("..")

import torch
import torch.nn as nn
from datasets import combined_brain_1mm, combined_brain_17mm, lidc
from generative.networks.nets import AutoencoderKL
from models.autoencoderkl import AutoencoderKLDownsampleControl
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse


def get_brain_datasets(root_dir, corrected=True):
    train_ds = combined_brain_1mm.CombinedBrainDataset(root_dir=root_dir, type="train", return_name=True, corrected=corrected)
    val_ds = combined_brain_1mm.CombinedBrainDataset(root_dir=root_dir, type="val", return_name=True, corrected=corrected)
    test_ds = combined_brain_1mm.CombinedBrainDataset(root_dir=root_dir, type="test", return_name=True, corrected=corrected)
    return train_ds, val_ds, test_ds


def get_lidc_datasets(root_dir):
    train_ds = lidc.LIDCDataset(root_dir=root_dir, type="train", return_name=True)
    val_ds = lidc.LIDCDataset(root_dir=root_dir, type="val", return_name=True)
    test_ds = lidc.LIDCDataset(root_dir=root_dir, type="test", return_name=True)
    return train_ds, val_ds, test_ds


def decode(args):    
    if args.dataset_type == "brain":
        train_ds, val_ds, test_ds = get_brain_datasets(args.root_dir)
    elif args.dataset_type == "lidc":
        train_ds, val_ds, test_ds = get_lidc_datasets(args.root_dir)
    
    config = OmegaConf.load(args.config_path)
    print(f"Loaded config from {args.config_path}")
    
    model = AutoencoderKLDownsampleControl(**config["stage1"]["params"])
    weight_path = args.weight_path
    output_path = args.outpath
    data = torch.load(weight_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(data["state_dict"])
    
    for i, _ in enumerate(tqdm(val_ds)):
        input = val_ds[i]["data"].unsqueeze(0).to(device)
        names = val_ds[i]["name"]
        with torch.no_grad():
            z_mu, z_sigma = model.module.encode(input)
        output = np.concatenate([z_mu.cpu().numpy().squeeze(), z_sigma.cpu().numpy().squeeze()])
        np.save(os.path.join(output_path, names+".npy"), output)
    
    for i, _ in enumerate(tqdm(train_ds)):
        input = train_ds[i]["data"].unsqueeze(0).to(device)
        names = train_ds[i]["name"]
        with torch.no_grad():
            z_mu, z_sigma = model.module.encode(input)
        output = np.concatenate([z_mu.cpu().numpy().squeeze(), z_sigma.cpu().numpy().squeeze()])
        np.save(os.path.join(output_path, names+".npy"), output)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, required=True)
    args.add_argument("--root_dir", type=str, required=True)
    args.add_argument("--dataset_type", type=str, required=True)
    args.add_argument("--weight_path", type=str, required=True)
    args.add_argument("--outpath", type=str, required=True)
    
    args = args.parse_args()
    
    print(args)

    decode(args)