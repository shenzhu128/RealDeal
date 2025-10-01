import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


def scale_to_zero_one(data):
    orig_min = data.min()
    orig_max = data.max()
    return (data - orig_min) / (orig_max - orig_min)


def scale_to_minusone_one(data):
    orig_min = data.min()
    orig_max = data.max()
    return ((data - orig_min) / (orig_max - orig_min)) * 2 - 1


class GeneratedDataset(Dataset):

    def __init__(
        self,
        root_dir,
        return_name=False,
        output_size=(176, 208, 184),
    ):
        self.root_dir = root_dir
        self.return_name = return_name
        self.output_size = output_size

        self.files = os.listdir(root_dir)
        self.files = [item for item in self.files if item.endswith(".nii.gz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.files[idx]
        img = nib.load(os.path.join(self.root_dir, file)).get_fdata()

        pad1 = (self.output_size[2] - img.shape[2]) // 2
        pad2 = (self.output_size[2] - img.shape[2]) - pad1
        pad3 = (self.output_size[1] - img.shape[1]) // 2
        pad4 = (self.output_size[1] - img.shape[1]) - pad3
        pad5 = (self.output_size[0] - img.shape[0]) // 2
        pad6 = (self.output_size[0] - img.shape[0]) - pad5

        imgout = torch.from_numpy(img).float()
        imgout = torch.unsqueeze(imgout, 0)
        imgout = F.pad(imgout, (pad1, pad2, pad3, pad4, pad5, pad6), mode="replicate")
        # imgout = scale_to_minusone_one(imgout)
        imgout = scale_to_zero_one(imgout)

        name = file.replace(".nii.gz", "")
        if self.return_name:
            # name = path.split("/")[-1].split(".")[0]
            return {"data": imgout, "name": name}
        return {"data": imgout}
