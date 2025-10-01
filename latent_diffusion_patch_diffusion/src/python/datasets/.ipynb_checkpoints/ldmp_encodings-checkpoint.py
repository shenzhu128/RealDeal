import glob
import os

import einops
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd 


def scale_to_zero_one(data):
    orig_min = data.min()
    orig_max = data.max()
    return (data - orig_min) / (orig_max - orig_min)


def scale_to_minusone_one(data):
    orig_min = data.min()
    orig_max = data.max()
    return ((data - orig_min) / (orig_max - orig_min)) * 2 - 1


class LDMPEncodingsMinusOneOneDataset(Dataset):
    def __init__(
        self,
        data_paths,
        type="train",
        return_name=False,
        transform=None,
        fixed_scale=0.2,
        output_shape=(48, 56, 48),
        if_sample=True,
    ):
        self.data_paths = data_paths
        data_list = os.listdir(self.data_paths)
        data_list = [item for item in data_list if item.endswith(".npy")]
        data_list = sorted(data_list)
        print(f"All data: {len(data_list)}")
        self.data_list = np.array(data_list)

        self.type = type
        self.return_name = return_name
        self.transform = transform
        # self.fixed_scale = fixed_scale
        self.output_shape = output_shape
        self.if_sample = if_sample

        if self.type == "val":
            self.data_list = self.data_list[:10]
        print(f"{self.type} data: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.data_list[idx]
        img = np.load(os.path.join(self.data_paths, file))
        img_mu = torch.from_numpy(img[:4]).float()
        img_sigma = torch.from_numpy(img[4:]).float()

        if self.if_sample:
            eps = torch.randn_like(img_sigma)
            img_mu = img_mu + eps * img_sigma

        pad1 = (self.output_shape[2] - img_mu.shape[3]) // 2
        pad2 = (self.output_shape[2] - img_mu.shape[3]) - pad1
        pad3 = (self.output_shape[1] - img_mu.shape[2]) // 2
        pad4 = (self.output_shape[1] - img_mu.shape[2]) - pad3
        pad5 = (self.output_shape[0] - img_mu.shape[1]) // 2
        pad6 = (self.output_shape[0] - img_mu.shape[1]) - pad5

        # img_mu = img_mu * self.fixed_scale
        img_mu = scale_to_minusone_one(img_mu)
        img_mu.unsqueeze(0)
        img_mu = F.pad(img_mu, (pad1, pad2, pad3, pad4, pad5, pad6), mode="reflect")

        name = file.replace(".npy", "")
        if self.return_name:
            return {
                "image": img_mu,
                "name": name,
            }
        return {"image": img_mu}


class LDMPEncodingsFixScaledDataset(Dataset):
    def __init__(
        self,
        data_paths,
        type="train",
        return_name=False,
        transform=None,
        fixed_scale=0.2,
        output_shape=(48, 56, 48),
        if_sample=True,
        ch=4,
    ):
        self.data_paths = data_paths
        data_list = os.listdir(self.data_paths)
        data_list = [item for item in data_list if item.endswith(".npy")]
        data_list = sorted(data_list)
        print(f"All data: {len(data_list)}")
        self.data_list = np.array(data_list)

        self.type = type
        self.return_name = return_name
        self.transform = transform
        self.fixed_scale = fixed_scale
        self.output_shape = output_shape
        self.if_sample = if_sample
        self.ch = ch

        if self.type == "val":
            self.data_list = self.data_list[:10]
        print(f"{self.type} data: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.data_list[idx]
        img = np.load(os.path.join(self.data_paths, file))
        img_mu = torch.from_numpy(img[:self.ch]).float()
        img_sigma = torch.from_numpy(img[self.ch:]).float()

        if self.if_sample:
            eps = torch.randn_like(img_sigma)
            img_mu = img_mu + eps * img_sigma

        pad1 = (self.output_shape[2] - img_mu.shape[3]) // 2
        pad2 = (self.output_shape[2] - img_mu.shape[3]) - pad1
        pad3 = (self.output_shape[1] - img_mu.shape[2]) // 2
        pad4 = (self.output_shape[1] - img_mu.shape[2]) - pad3
        pad5 = (self.output_shape[0] - img_mu.shape[1]) // 2
        pad6 = (self.output_shape[0] - img_mu.shape[1]) - pad5

        img_mu = img_mu * self.fixed_scale
        img_mu.unsqueeze(0)
        img_mu = F.pad(img_mu, (pad1, pad2, pad3, pad4, pad5, pad6), mode="reflect")

        name = file.replace(".npy", "")
        if self.return_name:
            return {
                "image": img_mu,
                "name": name,
            }
        return {"image": img_mu}


class EightMMEncodingsFixScaledDataset(Dataset):
    def __init__(
        self,
        data_paths,
        type="train",
        transform=None,
        fixed_scale=0.2,
        output_shape=(48, 56, 48, 3),
        if_sample=True,
        suffix=".nii.gz",
        split_file="/home/sz9jt/projects/generative_brain/unique_train_val_subjs_split_1.csv",
        scale_each_ch=False,
        ch1_fixed_scale=None,
        ch2_fixed_scale=None,
        ch3_fixed_scale=None,
    ):
        self.data_paths = data_paths
        self.split_file = pd.read_csv(split_file)

        data_list = list()
        for idx, row in self.split_file.iterrows():
            r = dict(row)
            s = f"{r['dataset_name']}.{r['subj_id']}.{r['session_id']}.{r['run_id']}_encoding.nii.gz"
            data_list.append(s)

        data_list = sorted(data_list)
        print(f"All data: {len(data_list)}")
        self.data_list = np.array(data_list)

        self.type = type
        self.transform = transform
        self.fixed_scale = fixed_scale
        self.output_shape = output_shape
        self.if_sample = if_sample
        self.suffix = suffix
        
        self.scale_each_ch = scale_each_ch
        self.ch1_fixed_scale = float(ch1_fixed_scale)
        self.ch2_fixed_scale = float(ch2_fixed_scale)
        self.ch3_fixed_scale = float(ch3_fixed_scale)
        
        self.ch_scale_tensor = torch.tensor([
            float(self.ch1_fixed_scale),
            float(self.ch2_fixed_scale),
            float(self.ch3_fixed_scale)
        ]).float()

        if self.type == "val":
            self.data_list = self.data_list[:10]
        print(f"{self.type} data: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.data_list[idx]
        img_mu = nib.load(os.path.join(self.data_paths, file)).get_fdata()
        img_mu = torch.from_numpy(img_mu).float()
        img_sigma = nib.load(os.path.join(self.data_paths, "sigma", file)).get_fdata()
        img_sigma = torch.from_numpy(img_sigma).float()

        if self.if_sample:
            eps = torch.randn_like(img_sigma)
            img_mu = img_mu + eps * img_sigma

        pad1 = (self.output_shape[2] - img_mu.shape[2]) // 2
        pad2 = (self.output_shape[2] - img_mu.shape[2]) - pad1
        pad3 = (self.output_shape[1] - img_mu.shape[1]) // 2
        pad4 = (self.output_shape[1] - img_mu.shape[1]) - pad3
        pad5 = (self.output_shape[0] - img_mu.shape[0]) // 2
        pad6 = (self.output_shape[0] - img_mu.shape[0]) - pad5
        
        # print(img_mu[:, :, :, 0].min(), img_mu[:, :, :, 0].max())
        # print(img_mu[:, :, :, 1].min(), img_mu[:, :, :, 1].max())
        # print(img_mu[:, :, :, 2].min(), img_mu[:, :, :, 2].max())
        
        if not self.scale_each_ch:
            img_mu = img_mu * self.fixed_scale
        else:
            img_mu.mul_(self.ch_scale_tensor.view(1, 1, 1, -1))
            # img_mu[:, :, :, 0] = img_mu[:, :, :, 0] * self.ch1_fixed_scale
            # img_mu[:, :, :, 1] = img_mu[:, :, :, 1] * self.ch2_fixed_scale
            # img_mu[:, :, :, 2] = img_mu[:, :, :, 2] * self.ch3_fixed_scale
            
#         print(img_mu[:, :, :, 0].min(), img_mu[:, :, :, 0].max())
#         print(img_mu[:, :, :, 1].min(), img_mu[:, :, :, 1].max())
#         print(img_mu[:, :, :, 2].min(), img_mu[:, :, :, 2].max())
#         print("===")
        
        img_mu = einops.rearrange(img_mu, "x y z d -> d x y z")
        img_mu = F.pad(img_mu, (pad1, pad2, pad3, pad4, pad5, pad6), mode="reflect")

        name = file.replace(self.suffix, "")
        return {
            "image": img_mu,
            "name": name,
        }


class EightMMEncodingsFixScaledLungDataset(Dataset):
    def __init__(
        self,
        data_paths,
        type="train",
        transform=None,
        fixed_scale=0.2,
        output_shape=(48, 56, 48, 3),
        if_sample=True,
        suffix=".nii.gz",
        # split_file="/home/sz9jt/projects/generative_brain/unique_train_val_subjs_split_1.csv"
    ):
        self.data_paths = data_paths
        # self.split_file = pd.read_csv(split_file)

        data_list = os.listdir(data_paths)
        data_list = [f for f in data_list if f.endswith(".nii.gz")]
        data_list = sorted(data_list)

        print(f"All data: {len(data_list)}")
        self.data_list = np.array(data_list)

        self.type = type
        self.transform = transform
        self.fixed_scale = fixed_scale
        self.output_shape = output_shape
        self.if_sample = if_sample
        self.suffix = suffix

        if self.type == "val":
            self.data_list = self.data_list[:10]
        print(f"{self.type} data: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.data_list[idx]
        img_mu = nib.load(os.path.join(self.data_paths, file)).get_fdata()
        img_mu = torch.from_numpy(img_mu).float()
        img_sigma = nib.load(os.path.join(self.data_paths, "sigma", file)).get_fdata()
        img_sigma = torch.from_numpy(img_sigma).float()

        if self.if_sample:
            eps = torch.randn_like(img_sigma)
            img_mu = img_mu + eps * img_sigma

        pad1 = (self.output_shape[2] - img_mu.shape[2]) // 2
        pad2 = (self.output_shape[2] - img_mu.shape[2]) - pad1
        pad3 = (self.output_shape[1] - img_mu.shape[1]) // 2
        pad4 = (self.output_shape[1] - img_mu.shape[1]) - pad3
        pad5 = (self.output_shape[0] - img_mu.shape[0]) // 2
        pad6 = (self.output_shape[0] - img_mu.shape[0]) - pad5

        img_mu = img_mu * self.fixed_scale
        img_mu = einops.rearrange(img_mu, "x y z d -> d x y z")
        img_mu = F.pad(img_mu, (pad1, pad2, pad3, pad4, pad5, pad6), mode="reflect")

        name = file.replace(self.suffix, "")
        return {
            "image": img_mu,
            "name": name,
        }