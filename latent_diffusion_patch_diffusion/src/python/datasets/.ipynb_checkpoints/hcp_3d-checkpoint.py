import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data.dataset import Dataset


def scale_to_zero_one(data):
    orig_min = data.min()
    orig_max = data.max()
    return (data - orig_min) / (orig_max - orig_min)


def scale_to_minusone_one(data):
    orig_min = data.min()
    orig_max = data.max()
    return ((data - orig_min) / (orig_max - orig_min)) * 2 - 1


class HCP3DDataset(Dataset):

    def __init__(
        self,
        root_dir,
        type="train",
        output_size=(224, 296, 224),
    ):
        print(f"Using root dir {root_dir}")
        spilt_path = Path("/home/sz9jt/data/HCP3D/split.csv")
        split_df = pd.read_csv(spilt_path)
        self.root_dir = Path(root_dir)
        self.type = type
        self.output_size = output_size

        print(f"All data: {len(split_df)}")
        if type == "train":
            split_df = split_df[split_df["split"] == "train"]
        elif type == "test":
            split_df = split_df[split_df["split"] == "test"]
        elif type == "all":
            pass
        else:
            raise ValueError("Unknown type: %s" % type)
        print(f"Dataset size: {len(split_df)}")

        self.files = split_df["scan"].values
        for item in self.files:
            if not os.path.exists(self.root_dir / item):
                print(item)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.root_dir / self.files[idx]
        img = nib.load(file_path).get_fdata()

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
        # imgout = scale_to_zero_one(imgout)

        return {"data": imgout, "name": self.files[idx]}


class HCP3DCondDataset(Dataset):
    def __init__(
        self,
        cond_img_path,
        gt_img_path,
        patch_size=(64, 64, 64),
        type="train",
    ):
        self.cond_img_path = Path(cond_img_path)
        self.gt_img_path = Path(gt_img_path)

        # make sure cond and gt image path match
        config = OmegaConf.load(self.cond_img_path / "../config.yaml")
        assert config.args.root_dir == gt_img_path

        spilt_path = Path("/home/sz9jt/data/HCP3D/split.csv")
        split_df = pd.read_csv(spilt_path)
        print(f"All data: {len(split_df)}")
        if type == "train":
            split_df = split_df[split_df["split"] == "train"]
        elif type == "test":
            split_df = split_df[split_df["split"] == "test"]
        elif type == "val":
            split_df = split_df[split_df["split"] == "test"][:2]
        elif type == "all":
            pass
        else:
            raise ValueError("Unknown type: %s" % type)
        print(f"Dataset size: {len(split_df)}")

        self.files = split_df["scan"].values
        for item in self.files:
            if not os.path.exists(self.cond_img_path / item.replace(".nii.gz", ".npy")):
                print(self.cond_img_path / item.replace(".nii.gz", ".npy"))
            if not os.path.exists(self.gt_img_path / item):
                print(self.gt_img_path / item)

        sample_noisy_img = nib.load(self.gt_img_path / item).get_fdata()
        sample_smooth_img = np.load(self.cond_img_path / item.replace(".nii.gz", ".npy"))
        print(
            "Sample noisy image data range:",
            sample_noisy_img.min(),
            sample_noisy_img.max(),
        )
        print(
            "Sample smooth image data range:",
            sample_smooth_img.min(),
            sample_smooth_img.max(),
        )

        # determine if patching is needed
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        self.patch_size = patch_size

        if self.patch_size == sample_noisy_img.shape:
            self.patch = False
        else:
            self.patch = True

        # # get mask
        # self.mask = np.load("/home/sz9jt/data/HCP3D/hcp3d_total_mask_cropped_binary.npy")
        # brain_voxels = np.argwhere(self.mask > 0)
        # self.min_bounds = brain_voxels.min(axis=0)  # min x, y, z
        # self.max_bounds = brain_voxels.max(axis=0)  # max x, y, z

        # # precompute all patches
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # fractions = torch.zeros(self.mask.shape).to(device)
        # print("Precomputing patches...")
        # num_patches = (
        #     (self.max_bounds[0] - self.patch_size[0] - self.min_bounds[0])
        #     * (self.max_bounds[1] - self.patch_size[1] - self.min_bounds[1])
        #     * (self.max_bounds[2] - self.patch_size[2] - self.min_bounds[2])
        # )
        # self.mask = torch.from_numpy(self.mask).to(device)
        # print(f"Number of patches: {num_patches}")
        # i = 0
        # for x in range(self.min_bounds[0], self.max_bounds[0] - self.patch_size[0]):
        #     for y in range(self.min_bounds[1], self.max_bounds[1] - self.patch_size[1]):
        #         for z in range(self.min_bounds[2], self.max_bounds[2] - self.patch_size[2]):
        #             i += 1
        #             patch_mask = self.mask[
        #                 x : x + self.patch_size[0],
        #                 y : y + self.patch_size[1],
        #                 z : z + self.patch_size[2],
        #             ]
        #             brain_fraction = torch.sum(patch_mask > 0) / patch_mask.numel()
        #             fractions[x, y, z] = brain_fraction
        #             if i % 10000 == 0:
        #                 print(f"Progress: {i}/{num_patches}")
        # fractions = fractions.cpu().numpy()
        # np.save("fractions.npy", fractions)

        self.fractions = np.load("/home/sz9jt/projects/generative_brain/src/python/datasets/fractions.npy")
        self.coords = np.argwhere(self.fractions > 0.5)
        print(self.coords.shape)
        print(f"Number of patches: {self.coords.shape[0]}")

    def __extract_patches__(self):
        rand_idx = np.random.randint(0, self.coords.shape[0])
        x, y, z = self.coords[rand_idx]
        return x, y, z

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = nib.load(self.gt_img_path / self.files[index]).get_fdata()
        cond_image = np.load(self.cond_img_path / (self.files[index].replace(".nii.gz", ".npy")))

        # patch or no
        if self.patch:
            x, y, z = self.__extract_patches__()
            img = img[
                x : x + self.patch_size[0],
                y : y + self.patch_size[1],
                z : z + self.patch_size[2],
            ]
            cond_image = cond_image[
                x : x + self.patch_size[0],
                y : y + self.patch_size[1],
                z : z + self.patch_size[2],
            ]

        img = img[None, ...]
        cond_image = cond_image[None, ...]

        ret = {}
        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["path"] = str(self.gt_img_path / self.files[index])
        return ret

    def __len__(self):
        return len(self.files)


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


class HCP3DDataset1mm(Dataset):

    def __init__(
        self,
        root_dir,
        type="train",
        output_size=(160, 208, 160),
    ):
        spilt_path = "/home/sz9jt/projects/generative_brain/src/splits/hcp_3d_split.csv"
        split_df = pd.read_csv(spilt_path, comment="#", index_col=None)
        split_df = split_df.astype(str)
        self.root_dir = root_dir
        self.type = type
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
            path = get_data_path_1mm(self.root_dir, *split_df[cols].iloc[i])
            self.data_paths.append(path)
            self.names.append(".".join(split_df[cols].iloc[i]))
        assert len(self.data_paths) == len(split_df)

        for item in self.data_paths:
            if not os.path.exists(item):
                print(item)

        self.numpy_paths = []
        for item in self.data_paths:
            tmp = Path(item).parent / "scaled_padded_data.npy"
            self.numpy_paths.append(tmp)
            if not os.path.exists(tmp):
                print(tmp)
        # self.data_paths_and_names = sorted(zip(self.data_paths, self.names), key=lambda x: x[0])
        # self.data_paths_and_names = np.array(self.data_paths_and_names)
        # self.data_paths, self.names = zip(*tmp)
        # self.data_paths = np.array(self.data_paths)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.numpy_paths[idx]
        name = self.names[idx]
        img = np.load(path)

        imgout = torch.from_numpy(img).float()
        imgout = torch.unsqueeze(imgout, 0)
        # numpy data is already cropped and scaled to -1 to 1

        return {"data": imgout, "name": name}


class TomsHCP3DEncodingsDataset(Dataset):
    """
    This dataset uses encodings from Tom's encoder decoder model
    Trained on HCP3D 0.7mm data, with only MSE loss
    """

    def __init__(
        self,
        root_dir,
        type="train",
        output_size=(4, 56, 72, 56),
        encoding_name="T1w_encoded_rescaled.npy",
    ):
        self.root_dir = Path(root_dir)
        split_df = pd.read_csv(self.root_dir / "hcp_split.csv")
        split_df = split_df.astype(str)

        self.type = type
        self.output_size = output_size
        print(f"All data: {len(split_df)}")
        if type == "train":
            split_df = split_df[split_df["split"] == "train"]
        elif type == "val":
            split_df = split_df[split_df["split"] == "train"][:2]
        elif type == "test":
            split_df = split_df[split_df["split"] == "test"]
        elif type == "all":
            pass
        else:
            raise ValueError("Unknown type: %s" % type)
        print(f"Dataset size: {len(split_df)}")

        self.data_paths = []
        self.names = []

        for i in range(len(split_df)):
            path = self.root_dir / split_df["subject"].iloc[i] / "T1w" / encoding_name
            self.data_paths.append(path)
            self.names.append(split_df["subject"].iloc[i])

        for item in self.data_paths:
            if not os.path.exists(item):
                print(item)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.data_paths[idx]
        name = self.names[idx]
        img = np.load(path).squeeze()
        img = torch.from_numpy(img).float()

        return {"image": img, "name": name, "path": str(path)}


class TomsHCP3DCondPatchDataset(Dataset):

    def __init__(
        self,
        root_dir,
        cond_img_name="T1w_reconstructed.nii.gz",
        gt_img_name="T1w_cropped.nii.gz",
        patch_size=(64, 64, 64),
        type="train",
        threshold=0.2,
    ):
        self.root_dir = Path(root_dir)
        split_df = pd.read_csv(self.root_dir / "hcp_split.csv")
        split_df = split_df.astype(str)

        self.type = type
        self.cond_img_name = cond_img_name
        self.gt_img_name = gt_img_name

        print(f"All data: {len(split_df)}")
        if type == "train":
            split_df = split_df[split_df["split"] == "train"]
        elif type == "val":
            split_df = split_df[split_df["split"] == "train"][:2]
        elif type == "test":
            split_df = split_df[split_df["split"] == "test"]
        elif type == "all":
            pass
        else:
            raise ValueError("Unknown type: %s" % type)
        print(f"Dataset size: {len(split_df)}")

        self.all_gt_paths = []
        self.all_cond_paths = []
        self.names = []

        for i in range(len(split_df)):
            gt_path = self.root_dir / split_df["subject"].iloc[i] / "T1w" / self.gt_img_name
            cond_path = self.root_dir / split_df["subject"].iloc[i] / "T1w" / self.cond_img_name
            self.all_gt_paths.append(gt_path)
            self.all_cond_paths.append(cond_path)
            self.names.append(split_df["subject"].iloc[i])

        for i in range(len(self.all_gt_paths)):
            if not os.path.exists(self.all_gt_paths[i]):
                print(self.all_gt_paths[i])
            if not os.path.exists(self.all_cond_paths[i]):
                print(self.all_cond_paths[i])

        # determine if patching is needed
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        self.patch_size = patch_size
        assert len(self.patch_size) == 3, "Patch size must be 3D"

        sample_gt_img = nib.load(self.all_gt_paths[0]).get_fdata()
        if self.patch_size == sample_gt_img.shape:
            self.patch = False
        else:
            self.patch = True

        if self.patch_size[0] == 64:
            self.fractions = np.load(self.root_dir / "fractions_64.npy")
            assert self.fractions.shape == sample_gt_img.shape, "Fractions shape mismatch"
            fraction_shape = self.fractions.shape
            self.fractions = self.fractions[
                0 : fraction_shape[0] - patch_size[0] + 1,
                0 : fraction_shape[1] - patch_size[1] + 1,
                0 : fraction_shape[2] - patch_size[2] + 1,
            ]
            # only use patches with brain fraction >= threshold
            self.coords = np.argwhere(self.fractions >= threshold)
            print(f"Number of patches: {self.coords.shape[0]} / {self.fractions.size}")
            print(f"Threshold: {threshold}")
            print(f"Percentage of patches >= threshold: {self.coords.shape[0] / (self.fractions.size) * 100:.2f}%")
        elif self.patch_size == (224, 288, 224):
            pass
        else:
            raise NotImplementedError("Only 64x64x64 patches are supported")

    def __extract_patches__(self):
        rand_idx = np.random.randint(0, self.coords.shape[0])
        x, y, z = self.coords[rand_idx]
        return x, y, z

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = nib.load(self.all_gt_paths[index]).get_fdata()
        cond_image = nib.load(self.all_cond_paths[index]).get_fdata()

        # patch or no
        if self.patch:
            x, y, z = self.__extract_patches__()
            img = img[
                x : x + self.patch_size[0],
                y : y + self.patch_size[1],
                z : z + self.patch_size[2],
            ]
            cond_image = cond_image[
                x : x + self.patch_size[0],
                y : y + self.patch_size[1],
                z : z + self.patch_size[2],
            ]

        img = img[None, ...]
        cond_image = cond_image[None, ...]

        ret = {}
        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["path"] = str(self.all_gt_paths[index])
        return ret

    def __len__(self):
        return len(self.names)


class TomsHCP3DCondSliceDataset(Dataset):

    def __init__(
        self,
        root_dir,
        cond_img_name="T1w_reconstructed.nii.gz",
        gt_img_name="T1w_cropped.nii.gz",
    ):
        self.root_dir = Path(root_dir)
        split_df = pd.read_csv(self.root_dir / "hcp_split.csv")
        split_df = split_df.astype(str)

        self.type = type
        self.cond_img_name = cond_img_name
        self.gt_img_name = gt_img_name

        print(f"All data: {len(split_df)}")
        if type == "train":
            split_df = split_df[split_df["split"] == "train"]
        elif type == "val":
            split_df = split_df[split_df["split"] == "train"][:2]
        elif type == "test":
            split_df = split_df[split_df["split"] == "test"]
        elif type == "all":
            pass
        else:
            raise ValueError("Unknown type: %s" % type)
        print(f"Dataset size: {len(split_df)}")

        self.all_gt_paths = []
        self.all_cond_paths = []
        self.names = []

        for i in range(len(split_df)):
            gt_path = self.root_dir / split_df["subject"].iloc[i] / "T1w" / self.gt_img_name
            cond_path = self.root_dir / split_df["subject"].iloc[i] / "T1w" / self.cond_img_name
            self.all_gt_paths.append(gt_path)
            self.all_cond_paths.append(cond_path)
            self.names.append(split_df["subject"].iloc[i])

        for i in range(len(self.all_gt_paths)):
            if not os.path.exists(self.all_gt_paths[i]):
                print(self.all_gt_paths[i])
            if not os.path.exists(self.all_cond_paths[i]):
                print(self.all_cond_paths[i])

        # determine if patching is needed
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        self.patch_size = patch_size
        assert len(self.patch_size) == 3, "Patch size must be 3D"

        sample_gt_img = nib.load(self.all_gt_paths[0]).get_fdata()
        if self.patch_size == sample_gt_img.shape:
            self.patch = False
        else:
            self.patch = True

        if self.patch_size[0] == 64:
            self.fractions = np.load(self.root_dir / "fractions_64.npy")
            assert self.fractions.shape == sample_gt_img.shape, "Fractions shape mismatch"
            fraction_shape = self.fractions.shape
            self.fractions = self.fractions[
                0 : fraction_shape[0] - patch_size[0] + 1,
                0 : fraction_shape[1] - patch_size[1] + 1,
                0 : fraction_shape[2] - patch_size[2] + 1,
            ]
            # only use patches with brain fraction >= threshold
            self.coords = np.argwhere(self.fractions >= threshold)
            print(f"Number of patches: {self.coords.shape[0]} / {self.fractions.size}")
            print(f"Threshold: {threshold}")
            print(f"Percentage of patches >= threshold: {self.coords.shape[0] / (self.fractions.size) * 100:.2f}%")
        elif self.patch_size == (224, 288, 224):
            pass
        else:
            raise NotImplementedError("Only 64x64x64 patches are supported")

    def __extract_patches__(self):
        rand_idx = np.random.randint(0, self.coords.shape[0])
        x, y, z = self.coords[rand_idx]
        return x, y, z

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = nib.load(self.all_gt_paths[index]).get_fdata()
        cond_image = nib.load(self.all_cond_paths[index]).get_fdata()

        # patch or no
        if self.patch:
            x, y, z = self.__extract_patches__()
            img = img[
                x : x + self.patch_size[0],
                y : y + self.patch_size[1],
                z : z + self.patch_size[2],
            ]
            cond_image = cond_image[
                x : x + self.patch_size[0],
                y : y + self.patch_size[1],
                z : z + self.patch_size[2],
            ]

        img = img[None, ...]
        cond_image = cond_image[None, ...]

        ret = {}
        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["path"] = str(self.all_gt_paths[index])
        return ret

    def __len__(self):
        return len(self.names)
