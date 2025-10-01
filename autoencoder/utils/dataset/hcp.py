import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import nibabel as nib
import glob

class HCPDataset(Dataset):
    def __init__(self, root_dir, filename, scale_factor = 1.0, random_crop = False):
        self.root_dir = root_dir
        self.filename = filename
        self.filelist = glob.glob(os.path.join(root_dir, "**/T1w/", filename))
        self.scale_factor = scale_factor
        self.random_crop = random_crop

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        path = self.filelist[index]
        imgout = 1
        img = nib.load(path).get_fdata()

        imgout = torch.from_numpy(img).float()

        mindim = min(imgout.shape)
        if self.random_crop > 0 and self.random_crop <= mindim:
            lowx = np.random.randint(low = 0, high = imgout.shape[0] - self.random_crop)
            lowy = np.random.randint(low = 0, high = imgout.shape[1] - self.random_crop)
            lowz = np.random.randint(low = 0, high = imgout.shape[2] - self.random_crop)
            imgout = imgout[lowx:(lowx + self.random_crop), lowy:(lowy + self.random_crop), lowz:(lowz + self.random_crop)]
        elif self.random_crop > mindim:
            print("Warning: not cropping, random_crop = ", self.random_crop, ", which is bigger than mindim = ", mindim)

        # Add axis
        imgout = torch.unsqueeze(imgout, 0)

        return {"data" : imgout, "subjID" : str(path)}
