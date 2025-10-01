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


def get_data_path_1mm(base_path, dataset_name, subj_id, session_id, run_id):
    # deal with empty session_id and run_id
    if session_id == "EMPTY":
        session_id = ""
    if run_id == "EMPTY":
        run_id = ""

    if dataset_name == "hcp":
        return os.path.join(
            base_path,
            dataset_name,
            "derivatives/acpc_align",
            subj_id,
            session_id,
            run_id,
            "acpc-align_1mm_interpolated/acpc_1mm_t1w_brain.nii.gz",
        )
    elif dataset_name == "abide1":
        return os.path.join(
            base_path,
            dataset_name,
            "derivatives/acpc_align2",
            subj_id,
            session_id,
            run_id,
            "acpc-align_1mm_interpolated/acpc_1mm_t1w_brain.nii.gz",
        )
    elif dataset_name == "abide2":
        return os.path.join(
            base_path,
            dataset_name,
            "derivatives/acpc_align2",
            subj_id,
            session_id,
            run_id,
            "acpc-align_1mm_interpolated/acpc_1mm_t1w_brain.nii.gz",
        )
    elif dataset_name == "oasis3":
        return os.path.join(
            base_path,
            dataset_name,
            "derivatives/acpc_align2",
            subj_id,
            session_id,
            run_id,
            "acpc-align_1mm_interpolated/acpc_1mm_t1w_brain.nii.gz",
        )
    elif dataset_name == "corr":
        return os.path.join(
            base_path,
            dataset_name,
            "derivatives/acpc_align2",
            subj_id,
            session_id,
            run_id,
            "acpc-align_1mm_interpolated/acpc_1mm_t1w_brain.nii.gz",
        )
    else:
        raise ValueError("Unknown dataset name: %s" % dataset_name)


class CombinedBrainDataset(Dataset):

    def __init__(
        self,
        root_dir,
        type="train",
        return_name=False,
        output_size=(192, 224, 192),
        # corrected=False,
    ):
        spilt_path = "/home/sz9jt/projects/generative_brain/src/splits/t1w_brain_2176340263_split_1.csv"
        split_df = pd.read_csv(spilt_path, comment="#", index_col=None)
        self.root_dir = root_dir
        self.type = type
        self.return_name = return_name
        self.output_size = output_size
        print(f"All data: {len(split_df)}")
        if type == "train":
            split_df = split_df[split_df["split"] == "train"]
        elif type == "val":
            split_df = split_df[split_df["split"] == "val"]
        elif type == "test":
            split_df = split_df[split_df["split"] == "test"]
        elif type == "train+val":
            split_df = split_df[split_df["split"] != "test"]
        elif type == "all":
            pass
        else:
            raise ValueError("Unknown type: %s" % type)
        print(f"Dataset size: {len(split_df)}")

        self.data_paths = []
        self.names = []
        cols = ["dataset_name", "subj_id", "session_id", "run_id"]
        for i in range(len(split_df)):
            # if corrected:
            #     path = get_corrected_data_path_1mm(self.root_dir, *split_df[cols].iloc[i])
            # else:
            #     path = get_data_path_1mm(self.root_dir, *split_df[cols].iloc[i])
            path = get_data_path_1mm(self.root_dir, *split_df[cols].iloc[i])
            self.data_paths.append(path)
            self.names.append(".".join(split_df[cols].iloc[i]))
        assert len(self.data_paths) == len(split_df)

        for item in self.data_paths:
            if not os.path.exists(item):
                print(item)
        self.data_paths_and_names = sorted(zip(self.data_paths, self.names), key=lambda x: x[0])
        self.data_paths_and_names = np.array(self.data_paths_and_names)
        # self.data_paths, self.names = zip(*tmp)
        # self.data_paths = np.array(self.data_paths)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path, name = self.data_paths_and_names[idx]
        img = nib.load(path).get_fdata()

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

        if self.return_name:
            # name = path.split("/")[-1].split(".")[0]
            return {"data": imgout, "name": name}
        return {"data": imgout}
