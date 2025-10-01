import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import argparse
import nibabel as nib
import glob
import os

from utils.dataset.hcp import HCPDataset
from utils.models.models3D import ResNetAutoencoder3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="base path for HCP data")
    parser.add_argument("--checkpoint", help="checkpoint from which to start")
    parser.add_argument(
        "--randseed",
        type=int,
        default=42,
        help="random seed for random train/test split",
    )
    args = parser.parse_args()

    # hcp_data = HCPDataset(args.data_path, "T1w_cropped.nii.gz")
    # num_data = len(hcp_data)

    # generator = torch.Generator().manual_seed(args.randseed)
    # num_train = (int)(0.8 * num_data)
    # num_test = num_data - num_train
    # print("Number of training data =", num_train)
    # print("Number of testing data =", num_test, flush = True)
    # [train_data, test_data] = random_split(hcp_data, [num_train, num_test], generator = generator)

    # train_loader = DataLoader(train_data, batch_size = num_train, shuffle = True, num_workers = 4)
    # data = next(iter(train_loader))
    # np.save("trainIDs.npy", data["subjID"])

    filelist = glob.glob(os.path.join(args.data_path, "**/T1w/", "T1w_encoded.npy"))
    model = ResNetAutoencoder3d(num_channels=64, nonlinearity=F.leaky_relu).to(device)
    if args.checkpoint != None:
        print("Loading checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
    model.eval()

    nifti_tmp = nib.load("/data/HCP_preprocess/100206/T1w/T1w_cropped.nii.gz")
    affmat = nifti_tmp.affine
    print(affmat)

    i = 0
    for filename in filelist:
        znp = np.load(filename)

        z = torch.from_numpy(znp).to(device)

        x = model.decoder(z)

        outfile = filename.replace("T1w_encoded.npy", "T1w_reconstructed.nii.gz")

        img = nib.Nifti1Image(np.squeeze(x.data.cpu().numpy()), affine=affmat)
        nib.save(img, outfile)

        print(i)
        i = i + 1

        del x
        del z
