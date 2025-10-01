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


def bbox2mask(img_shape, bbox, dtype="uint8"):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, d, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """
    # bbox (left, posterior, inferior, height, width, depth)
    height, width, depth = img_shape[:3]

    mask = np.zeros((height, width, depth, 1), dtype=dtype)
    mask[bbox[0] : bbox[0] + bbox[3], bbox[1] : bbox[1] + bbox[4], bbox[2] : bbox[2] + bbox[5], :] = 1

    return mask


def cropping_bbox_dir(img_shape=(64, 64, 64), mask_direction="left", random_direction=False):
    # x     0 left      x right
    # y     0 posterior y anterior
    # z     0 inferior  z superior
    left_right, posterior_anterior, inferior_superior = img_shape
    all_directions = ["left", "right", "posterior", "anterior", "inferior", "superior"]
    if random_direction:
        mask_direction = np.random.choice(all_directions)
        # print("Using random direction", mask_direction)
    if mask_direction == "left":
        left, posterior, inferior, height, width, depth = (
            0,
            0,
            0,
            left_right // 2,
            posterior_anterior,
            inferior_superior,
        )
    elif mask_direction == "right":
        left, posterior, inferior, height, width, depth = (
            left_right // 2,
            0,
            0,
            left_right // 2,
            posterior_anterior,
            inferior_superior,
        )
    elif mask_direction == "posterior":
        left, posterior, inferior, height, width, depth = (
            0,
            0,
            0,
            left_right,
            posterior_anterior // 2,
            inferior_superior,
        )
    elif mask_direction == "anterior":
        left, posterior, inferior, height, width, depth = (
            0,
            posterior_anterior // 2,
            0,
            left_right,
            posterior_anterior // 2,
            inferior_superior,
        )
    elif mask_direction == "inferior":
        left, posterior, inferior, height, width, depth = (
            0,
            0,
            0,
            left_right,
            posterior_anterior,
            inferior_superior // 2,
        )
    elif mask_direction == "superior":
        left, posterior, inferior, height, width, depth = (
            0,
            0,
            inferior_superior // 2,
            left_right,
            posterior_anterior,
            inferior_superior // 2,
        )
    else:
        raise ValueError(
            f"mask direction {mask_direction} is not supported, please use left, right, posterior, anterior, inferior, superior"
        )
    return (left, posterior, inferior, height, width, depth)


class TomsHCP3DTrainPatchUncroppingDataset(Dataset):
    def __init__(
        self,
        root_dir,
        patch_size=(64, 64, 64),
        full_mask_prob=0.0,
        type="train",
        finetune=False,
    ):
        print(f"Using root dir {root_dir}")
        self.root_dir = Path(root_dir)
        spilt_path = self.root_dir / "hcp_split.csv"
        split_df = pd.read_csv(spilt_path).astype(str)

        self.full_mask_prob = full_mask_prob
        print(f"Full mask prob: {self.full_mask_prob}")
        self.mask_mode = "onedirection"

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

        self.all_subjects = split_df["subject"].values
        # for item in self.all_subjects:
        #     if not os.path.exists(self.root_dir / f"{item}/T1w/T1w_cropped.nii.gz"):
        #         print(self.root_dir / f"{item}/T1w/T1w_cropped.nii.gz", "does not exist")
        #     if not os.path.exists(self.root_dir / f"{item}/T1w/T1w_reconstructed.nii.gz"):
        #         print(self.root_dir / f"{item}/T1w/T1w_reconstructed.nii.gz", "does not exist")

        item = self.all_subjects[0]
        sample_noisy_img = nib.load(self.root_dir / item / "T1w/T1w_cropped.nii.gz").get_fdata()
        sample_smooth_img = nib.load(self.root_dir / item / "T1w/T1w_reconstructed.nii.gz").get_fdata()
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

        # keep original shape for normalizing coordinates
        self.original_shape = sample_noisy_img.shape

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

        self.fractions = np.load(self.root_dir / "fractions_64.npy")
        self.fractions = self.fractions[
            0 : self.fractions.shape[0] - patch_size[0] + 1,
            0 : self.fractions.shape[1] - patch_size[1] + 1,
            0 : self.fractions.shape[2] - patch_size[2] + 1,
        ]
        # get all coordinates
        self.coords = np.argwhere(self.fractions >= 0.0)
        print(f"Number of patches: {self.coords.shape[0]} / {self.fractions.size}")
        print(f"Percentage of patches >= threshold: {self.coords.shape[0] / (self.fractions.size) * 100:.2f}%")

        # make bins for finetuning patch selection
        # Define the number of bins or bin edges
        self.finetune = finetune
        self.num_bins = 10  # You can adjust this based on your needs
        bin_edges = np.linspace(0, 0.9, self.num_bins)  # Bins for fractions in range [0, 0.9)
        # Bin the fraction values
        bin_indices = (
            np.digitize(self.fractions[self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]], bin_edges) - 1
        )
        print(f"Bin numbers: {self.num_bins}")
        print(f"All bin indices: {np.unique(bin_indices)}, num bins: {len(np.unique(bin_indices))}")
        # Group coordinates by bin
        bins = {i: [] for i in range(self.num_bins)}
        for coord, bin_idx in zip(self.coords, bin_indices):
            bins[bin_idx].append(coord)
        for k, v in bins.items():
            if len(v) == 0:
                print(f"Bin {k} is empty")
        # Remove empty bins
        self.bins = {k: v for k, v in bins.items() if len(v) > 0}
        total_coords = 0
        for k, v in self.bins.items():
            print(f"Bin {k} has {len(v)} coordinates")
            total_coords += len(v)
        print(f"Total coordinates: {total_coords} / {self.coords.shape[0]}")
        if self.finetune:
            print(f"Using finetune mode")
        else:
            print(f"Using normal mode")

    def __extract_patches__(self):
        rand_idx = np.random.randint(0, self.coords.shape[0])
        x, y, z = self.coords[rand_idx]
        return x, y, z

    def __extract_patches_for_finetune__(self):
        # Randomly pick a bin
        selected_bin_idx = np.random.choice(self.num_bins)
        # Randomly pick a coordinate from the selected bin
        selected_bin = self.bins[selected_bin_idx]
        selected_coord = selected_bin[np.random.randint(0, len(selected_bin))]
        x, y, z = selected_coord

        return x, y, z

    def __len__(self):
        return len(self.all_subjects)

    def get_mask(self):
        if self.mask_mode == "manual":
            raise NotImplementedError("Manual mask generation is not implemented yet.")
            # mask = bbox2mask(self.patch_size, self.mask_config["shape"])
        elif self.mask_mode == "fourdirection" or self.mask_mode == "onedirection":
            mask = bbox2mask(
                self.patch_size,
                cropping_bbox_dir(
                    img_shape=self.patch_size,
                    random_direction=True,
                ),
            )
        elif self.mask_mode == "hybrid":
            raise NotImplementedError("Hybrid mask generation is not implemented yet.")
        elif self.mask_mode == "file":
            pass
        else:
            raise NotImplementedError(f"Mask mode {self.mask_mode} has not been implemented.")
        return mask.squeeze()

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = nib.load(self.root_dir / self.all_subjects[index] / "T1w/T1w_cropped.nii.gz").get_fdata()
        cond_image = nib.load(self.root_dir / self.all_subjects[index] / "T1w/T1w_reconstructed.nii.gz").get_fdata()

        # patch or no
        if self.patch:
            # if finetune, use the binning method
            if self.finetune:
                x, y, z = self.__extract_patches_for_finetune__()
            else:
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

        # get mask
        if np.random.rand() < self.full_mask_prob:
            mask = np.ones_like(img)
            # print("Using full mask")
            # print(np.unique(mask))
        else:
            mask = self.get_mask()
            mask = mask[None, ...]

        masked_cond_img = img * (1.0 - mask) + mask * np.random.randn(*img.shape)
        all_cond_img = [cond_image, masked_cond_img]
        all_cond_img = np.concatenate(all_cond_img, axis=0)

        coordinates = np.array([x, y, z])
        # print("Coordinates:", coordinates)
        # print("fractions shape:", self.fractions.shape)
        # print("Original shape:", self.original_shape)
        # rescale to -1 to 1
        coordinates = coordinates * 2 / self.original_shape - 1
        # print("Rescaled coordinates:", coordinates)

        ret = {}
        ret["gt_image"] = img.astype(np.float32)
        ret["cond_image"] = all_cond_img.astype(np.float32)
        ret["mask"] = mask.astype(np.float32)
        ret["path"] = str(self.root_dir / self.all_subjects[index])
        ret["coordinates"] = np.array(coordinates)[None, ...].astype(np.float32)
        ret["patch_size"] = np.array(self.patch_size)

        return ret


class TomsHCP3DTestPatchUncroppingWithoutCoordinatesDataset(Dataset):
    def __init__(
        self,
        root_dir,
        patch_size=(64, 64, 64),
        full_mask_prob=0.0,
        type="test",
    ):
        print(f"Using root dir {root_dir}")
        self.root_dir = Path(root_dir)
        spilt_path = self.root_dir / "hcp_split.csv"
        split_df = pd.read_csv(spilt_path).astype(str)

        self.full_mask_prob = full_mask_prob
        self.mask_mode = "onedirection"

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

        self.all_subjects = split_df["subject"].values
        for item in self.all_subjects:
            if not os.path.exists(self.root_dir / f"{item}/T1w/T1w_cropped.nii.gz"):
                print(self.root_dir / f"{item}/T1w/T1w_cropped.nii.gz", "does not exist")
            if not os.path.exists(self.root_dir / f"{item}/T1w/T1w_reconstructed.nii.gz"):
                print(self.root_dir / f"{item}/T1w/T1w_reconstructed.nii.gz", "does not exist")

        sample_noisy_img = nib.load(self.root_dir / item / "T1w/T1w_cropped.nii.gz").get_fdata()
        self.file_header = nib.load(self.root_dir / item / "T1w/T1w_cropped.nii.gz").header
        self.file_affine = nib.load(self.root_dir / item / "T1w/T1w_cropped.nii.gz").affine
        sample_smooth_img = nib.load(self.root_dir / item / "T1w/T1w_reconstructed.nii.gz").get_fdata()
        print("Sample noisy image data range:")
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

    def __len__(self):
        return len(self.all_subjects)

    def get_mask(self, direction="posterior"):
        if self.mask_mode == "manual":
            raise NotImplementedError("Manual mask generation is not implemented yet.")
            # mask = bbox2mask(self.patch_size, self.mask_config["shape"])
        elif self.mask_mode == "fourdirection" or self.mask_mode == "onedirection":
            mask = bbox2mask(
                self.patch_size,
                cropping_bbox_dir(
                    img_shape=self.patch_size,
                    mask_direction=direction,
                    random_direction=False,
                ),
            )
        elif self.mask_mode == "hybrid":
            raise NotImplementedError("Hybrid mask generation is not implemented yet.")
        elif self.mask_mode == "file":
            pass
        else:
            raise NotImplementedError(f"Mask mode {self.mask_mode} has not been implemented.")
        return mask.squeeze()

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = nib.load(self.root_dir / self.all_subjects[index] / "T1w/T1w_cropped.nii.gz").get_fdata()
        blurry_cond_img = nib.load(
            self.root_dir / self.all_subjects[index] / "T1w/T1w_reconstructed.nii.gz"
        ).get_fdata()
        # pad images to (224, 320, 256) (7 * 32, 10 * 32, 8 * 32)
        img = np.pad(img, ((0, 0), (16, 16), (16, 16)), mode="constant", constant_values=-1.0)
        blurry_cond_img = np.pad(blurry_cond_img, ((0, 0), (16, 16), (16, 16)), mode="constant", constant_values=-1.0)

        # determine number of patches at each dimension
        xdim_num_patches = img.shape[0] // self.patch_size[0]
        ydim_num_patches = img.shape[1] // self.patch_size[1]
        zdim_num_patches = img.shape[2] // self.patch_size[2]
        print("xdim_num_patches", xdim_num_patches)
        print("ydim_num_patches", ydim_num_patches)
        print("zdim_num_patches", zdim_num_patches)

        # initial_patch in the center
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        initial_patch_orig_img = img[
            starting_x : starting_x + self.patch_size[0],
            starting_y : starting_y + self.patch_size[1],
            starting_z : starting_z + self.patch_size[2],
        ]
        initial_patch_blurry_cond_img = blurry_cond_img[
            starting_x : starting_x + self.patch_size[0],
            starting_y : starting_y + self.patch_size[1],
            starting_z : starting_z + self.patch_size[2],
        ]
        initial_mask = np.ones_like(initial_patch_orig_img)
        initial_masked_cond_img = initial_patch_orig_img * (1.0 - initial_mask) + initial_mask * np.random.randn(
            *initial_patch_orig_img.shape
        )
        initial_all_cond_img = [initial_patch_blurry_cond_img, initial_masked_cond_img]

        ret = {}
        ret["subject_id"] = self.all_subjects[index]
        ret["file_header"] = self.file_header
        ret["file_affine"] = self.file_affine
        ret["complete_gt_image"] = img
        ret["complete_cond_image"] = blurry_cond_img

        ret["initial_patch_gt_image"] = np.array(initial_patch_orig_img).astype(np.float32)
        ret["initial_patch_cond_image"] = np.array(initial_all_cond_img).astype(np.float32)
        ret["initial_patch_mask"] = np.array(initial_mask).astype(np.float32)
        ret["initial_patch_coordinates"] = np.array([starting_x, starting_y, starting_z])

        # x     0 left      x right
        # y     0 posterior y anterior
        # z     0 inferior  z superior
        # go posterior (y goes down)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_y = starting_y - self.patch_size[1] // 2
        go_posterior_all_orig_imgs = []
        go_posterior_all_cond_imgs = []
        go_posterior_all_masks = []
        go_posterior_all_coordinates = []
        while starting_y >= 0:
            patch_orig_img = img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            patch_blurry_cond_img = blurry_cond_img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            # use a posterior mask when going posterior
            patch_mask = self.get_mask(direction="posterior")
            patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                *patch_orig_img.shape
            )
            patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
            patch_all_cond_img = np.array(patch_all_cond_img)
            go_posterior_all_orig_imgs.append(patch_orig_img)
            go_posterior_all_cond_imgs.append(patch_all_cond_img)
            go_posterior_all_masks.append(patch_mask)
            go_posterior_all_coordinates.append([starting_x, starting_y, starting_z])
            starting_y = starting_y - self.patch_size[1] // 2
        ret["go_posterior_gt_image"] = np.array(go_posterior_all_orig_imgs).astype(np.float32)
        ret["go_posterior_cond_image"] = np.array(go_posterior_all_cond_imgs).astype(np.float32)
        ret["go_posterior_mask"] = np.array(go_posterior_all_masks).astype(np.float32)
        ret["go_posterior_coordinates"] = np.array(go_posterior_all_coordinates)

        # go anterior (y goes up)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_y = starting_y + self.patch_size[1] // 2
        go_anterior_all_orig_imgs = []
        go_anterior_all_cond_imgs = []
        go_anterior_all_masks = []
        go_anterior_all_coordinates = []
        while starting_y + self.patch_size[1] <= img.shape[1]:
            patch_orig_img = img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            patch_blurry_cond_img = blurry_cond_img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            # use an anterior mask when going anterior
            patch_mask = self.get_mask(direction="anterior")
            patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                *patch_orig_img.shape
            )
            patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
            patch_all_cond_img = np.array(patch_all_cond_img)
            go_anterior_all_orig_imgs.append(patch_orig_img)
            go_anterior_all_cond_imgs.append(patch_all_cond_img)
            go_anterior_all_masks.append(patch_mask)
            go_anterior_all_coordinates.append([starting_x, starting_y, starting_z])
            starting_y = starting_y + self.patch_size[1] // 2
        ret["go_anterior_gt_image"] = np.array(go_anterior_all_orig_imgs).astype(np.float32)
        ret["go_anterior_cond_image"] = np.array(go_anterior_all_cond_imgs).astype(np.float32)
        ret["go_anterior_mask"] = np.array(go_anterior_all_masks).astype(np.float32)
        ret["go_anterior_coordinates"] = np.array(go_anterior_all_coordinates)

        # after that, we use a batch of 5 patches to go inferior - z goes down
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_z = starting_z - self.patch_size[2] // 2
        go_inferior_all_orig_imgs = []
        go_inferior_all_cond_imgs = []
        go_inferior_all_masks = []
        go_inferior_all_coordinates = []
        while starting_z >= 0:
            go_inferior_column_orig_imgs = []
            go_inferior_column_cond_imgs = []
            go_inferior_column_masks = []
            go_inferior_column_coordinates = []
            for i in range(ydim_num_patches):
                starting_y = i * self.patch_size[1]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go inferior, we want to recover the inferior part of the image
                # thus use an inferior mask
                patch_mask = self.get_mask(direction="inferior")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)
                go_inferior_column_orig_imgs.append(patch_orig_img)
                go_inferior_column_cond_imgs.append(patch_all_cond_img)
                go_inferior_column_masks.append(patch_mask)
                go_inferior_column_coordinates.append([starting_x, starting_y, starting_z])
            # add the column to the list
            go_inferior_all_orig_imgs.append(go_inferior_column_orig_imgs)
            go_inferior_all_cond_imgs.append(go_inferior_column_cond_imgs)
            go_inferior_all_masks.append(go_inferior_column_masks)
            go_inferior_all_coordinates.append(go_inferior_column_coordinates)
            starting_z = starting_z - self.patch_size[2] // 2
        ret["go_inferior_gt_image"] = np.array(go_inferior_all_orig_imgs).astype(np.float32)
        ret["go_inferior_cond_image"] = np.array(go_inferior_all_cond_imgs).astype(np.float32)
        ret["go_inferior_mask"] = np.array(go_inferior_all_masks).astype(np.float32)
        ret["go_inferior_coordinates"] = np.array(go_inferior_all_coordinates)

        # we also need to go superior (z goes up)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_z = starting_z + self.patch_size[2] // 2
        go_superior_all_orig_imgs = []
        go_superior_all_cond_imgs = []
        go_superior_all_masks = []
        go_superior_all_coordinates = []
        while starting_z + self.patch_size[2] <= img.shape[2]:
            go_superior_column_orig_imgs = []
            go_superior_column_cond_imgs = []
            go_superior_column_masks = []
            go_superior_column_coordinates = []
            for i in range(ydim_num_patches):
                starting_y = i * self.patch_size[1]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go superior, we want to recover the superior part of the image
                # thus use a superior mask
                patch_mask = self.get_mask(direction="superior")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)
                go_superior_column_orig_imgs.append(patch_orig_img)
                go_superior_column_cond_imgs.append(patch_all_cond_img)
                go_superior_column_masks.append(patch_mask)
                go_superior_column_coordinates.append([starting_x, starting_y, starting_z])
            # add the column to the list
            go_superior_all_orig_imgs.append(go_superior_column_orig_imgs)
            go_superior_all_cond_imgs.append(go_superior_column_cond_imgs)
            go_superior_all_masks.append(go_superior_column_masks)
            go_superior_all_coordinates.append(go_superior_column_coordinates)
            starting_z = starting_z + self.patch_size[2] // 2
        ret["go_superior_gt_image"] = np.array(go_superior_all_orig_imgs).astype(np.float32)
        ret["go_superior_cond_image"] = np.array(go_superior_all_cond_imgs).astype(np.float32)
        ret["go_superior_mask"] = np.array(go_superior_all_masks).astype(np.float32)
        ret["go_superior_coordinates"] = np.array(go_superior_all_coordinates)

        # after that, we use a batch of ydim_num_patches * zdim_num_patches patches
        # to go left and right
        # x     0 left      x right
        # y     0 posterior y anterior
        # z     0 inferior  z superior
        # first we go left (x goes down)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_x = starting_x - self.patch_size[0] // 2
        go_left_all_orig_imgs = []
        go_left_all_cond_imgs = []
        go_left_all_masks = []
        go_left_all_coordinates = []

        while starting_x >= 0:
            go_left_slab_orig_imgs = []
            go_left_slab_cond_imgs = []
            go_left_slab_masks = []
            go_left_slab_coordinates = []
            # for each y and z, we get a patch
            for i in range(ydim_num_patches * zdim_num_patches):
                nrow = i // zdim_num_patches  # each row has zdim_num_patches
                ncol = i % zdim_num_patches  # each column has ydim_num_patches
                starting_y = nrow * self.patch_size[1]
                starting_z = ncol * self.patch_size[2]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go left, we want to recover the left part of the image
                # thus use a left mask
                patch_mask = self.get_mask(direction="left")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)
                go_left_slab_orig_imgs.append(patch_orig_img)
                go_left_slab_cond_imgs.append(patch_all_cond_img)
                go_left_slab_masks.append(patch_mask)
                go_left_slab_coordinates.append([starting_x, starting_y, starting_z])
            # add the slab to the list
            go_left_all_orig_imgs.append(go_left_slab_orig_imgs)
            go_left_all_cond_imgs.append(go_left_slab_cond_imgs)
            go_left_all_masks.append(go_left_slab_masks)
            go_left_all_coordinates.append(go_left_slab_coordinates)
            starting_x = starting_x - self.patch_size[0] // 2
        ret["go_left_gt_image"] = np.array(go_left_all_orig_imgs).astype(np.float32)
        ret["go_left_cond_image"] = np.array(go_left_all_cond_imgs).astype(np.float32)
        ret["go_left_mask"] = np.array(go_left_all_masks).astype(np.float32)
        ret["go_left_coordinates"] = np.array(go_left_all_coordinates)

        # now we go right (x goes up)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_x = starting_x + self.patch_size[0] // 2
        go_right_all_orig_imgs = []
        go_right_all_cond_imgs = []
        go_right_all_masks = []
        go_right_all_coordinates = []

        while starting_x + self.patch_size[0] <= img.shape[0]:
            go_right_slab_orig_imgs = []
            go_right_slab_cond_imgs = []
            go_right_slab_masks = []
            go_right_slab_coordinates = []
            # for each y and z, we get a patch
            for i in range(ydim_num_patches * zdim_num_patches):
                nrow = i // zdim_num_patches
                ncol = i % zdim_num_patches
                starting_y = nrow * self.patch_size[1]
                starting_z = ncol * self.patch_size[2]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go right, we want to recover the right part of the image
                # thus use a right mask
                patch_mask = self.get_mask(direction="right")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)

                go_right_slab_orig_imgs.append(patch_orig_img)
                go_right_slab_cond_imgs.append(patch_all_cond_img)
                go_right_slab_masks.append(patch_mask)
                go_right_slab_coordinates.append([starting_x, starting_y, starting_z])
            # add the slab to the list
            go_right_all_orig_imgs.append(go_right_slab_orig_imgs)
            go_right_all_cond_imgs.append(go_right_slab_cond_imgs)
            go_right_all_masks.append(go_right_slab_masks)
            go_right_all_coordinates.append(go_right_slab_coordinates)
            starting_x = starting_x + self.patch_size[0] // 2
        ret["go_right_gt_image"] = np.array(go_right_all_orig_imgs).astype(np.float32)
        ret["go_right_cond_image"] = np.array(go_right_all_cond_imgs).astype(np.float32)
        ret["go_right_mask"] = np.array(go_right_all_masks).astype(np.float32)
        ret["go_right_coordinates"] = np.array(go_right_all_coordinates)

        return ret


class TomsHCP3DTestPatchUncroppingWithCoordinatesDataset(Dataset):
    def __init__(
        self,
        root_dir,
        patch_size=(64, 64, 64),
        full_mask_prob=0.0,
        type="test",
    ):
        print(f"Using root dir {root_dir}")
        self.root_dir = Path(root_dir)
        spilt_path = self.root_dir / "hcp_split.csv"
        split_df = pd.read_csv(spilt_path).astype(str)

        self.full_mask_prob = full_mask_prob
        self.mask_mode = "onedirection"

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

        self.all_subjects = split_df["subject"].values
        for item in self.all_subjects:
            if not os.path.exists(self.root_dir / f"{item}/T1w/T1w_cropped.nii.gz"):
                print(self.root_dir / f"{item}/T1w/T1w_cropped.nii.gz", "does not exist")
            if not os.path.exists(self.root_dir / f"{item}/T1w/T1w_reconstructed.nii.gz"):
                print(self.root_dir / f"{item}/T1w/T1w_reconstructed.nii.gz", "does not exist")

        sample_noisy_img = nib.load(self.root_dir / item / "T1w/T1w_cropped.nii.gz").get_fdata()
        sample_smooth_img = nib.load(self.root_dir / item / "T1w/T1w_reconstructed.nii.gz").get_fdata()
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
        self.original_shape = sample_noisy_img.shape
        print(f"Original shape: {self.original_shape}")

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

    def __len__(self):
        return len(self.all_subjects)

    def get_mask(self, direction="posterior"):
        if self.mask_mode == "manual":
            raise NotImplementedError("Manual mask generation is not implemented yet.")
            # mask = bbox2mask(self.patch_size, self.mask_config["shape"])
        elif self.mask_mode == "fourdirection" or self.mask_mode == "onedirection":
            mask = bbox2mask(
                self.patch_size,
                cropping_bbox_dir(
                    img_shape=self.patch_size,
                    mask_direction=direction,
                    random_direction=False,
                ),
            )
        elif self.mask_mode == "hybrid":
            raise NotImplementedError("Hybrid mask generation is not implemented yet.")
        elif self.mask_mode == "file":
            pass
        else:
            raise NotImplementedError(f"Mask mode {self.mask_mode} has not been implemented.")
        return mask.squeeze()

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = nib.load(self.root_dir / self.all_subjects[index] / "T1w/T1w_cropped.nii.gz").get_fdata()
        blurry_cond_img = nib.load(
            self.root_dir / self.all_subjects[index] / "T1w/T1w_reconstructed.nii.gz"
        ).get_fdata()
        # pad images to (224, 320, 256) (7 * 32, 10 * 32, 8 * 32)
        # only pad the y, z at the end (0, 32)
        img = np.pad(img, ((0, 0), (0, 32), (0, 32)), mode="constant", constant_values=-1.0)
        blurry_cond_img = np.pad(blurry_cond_img, ((0, 0), (0, 32), (0, 32)), mode="constant", constant_values=-1.0)

        # determine number of patches at each dimension
        xdim_num_patches = img.shape[0] // self.patch_size[0]
        ydim_num_patches = img.shape[1] // self.patch_size[1]
        zdim_num_patches = img.shape[2] // self.patch_size[2]
        print("xdim_num_patches", xdim_num_patches)
        print("ydim_num_patches", ydim_num_patches)
        print("zdim_num_patches", zdim_num_patches)

        # initial_patch in the center
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        initial_patch_orig_img = img[
            starting_x : starting_x + self.patch_size[0],
            starting_y : starting_y + self.patch_size[1],
            starting_z : starting_z + self.patch_size[2],
        ]
        initial_patch_blurry_cond_img = blurry_cond_img[
            starting_x : starting_x + self.patch_size[0],
            starting_y : starting_y + self.patch_size[1],
            starting_z : starting_z + self.patch_size[2],
        ]
        initial_mask = np.ones_like(initial_patch_orig_img)
        initial_masked_cond_img = initial_patch_orig_img * (1.0 - initial_mask) + initial_mask * np.random.randn(
            *initial_patch_orig_img.shape
        )
        initial_all_cond_img = [initial_patch_blurry_cond_img, initial_masked_cond_img]

        ret = {}
        ret["complete_gt_image"] = img
        ret["complete_cond_image"] = blurry_cond_img

        ret["initial_patch_gt_image"] = np.array(initial_patch_orig_img).astype(np.float32)
        ret["initial_patch_cond_image"] = np.array(initial_all_cond_img).astype(np.float32)
        ret["initial_patch_mask"] = np.array(initial_mask).astype(np.float32)
        # during training the coordinates are normalized to [-1, 1]
        # using the original shape
        # to avoid distribution shift, we also perform the same normalization
        tmp = np.array([starting_x, starting_y, starting_z])
        normalized_tmp = tmp * 2 / self.original_shape - 1
        ret["initial_patch_coordinates"] = tmp
        ret["initial_patch_coordinates_normalized"] = normalized_tmp

        # x     0 left      x right
        # y     0 posterior y anterior
        # z     0 inferior  z superior
        # go posterior (y goes down)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_y = starting_y - self.patch_size[1] // 2
        go_posterior_all_orig_imgs = []
        go_posterior_all_cond_imgs = []
        go_posterior_all_masks = []
        go_posterior_all_coordinates = []
        go_posterior_all_coordinates_normalized = []
        while starting_y >= 0:
            patch_orig_img = img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            patch_blurry_cond_img = blurry_cond_img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            # use a posterior mask when going posterior
            patch_mask = self.get_mask(direction="posterior")
            patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                *patch_orig_img.shape
            )
            patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
            patch_all_cond_img = np.array(patch_all_cond_img)
            go_posterior_all_orig_imgs.append(patch_orig_img)
            go_posterior_all_cond_imgs.append(patch_all_cond_img)
            go_posterior_all_masks.append(patch_mask)
            tmp = np.array([starting_x, starting_y, starting_z])
            normalized_tmp = tmp * 2 / self.original_shape - 1
            go_posterior_all_coordinates.append(tmp)
            go_posterior_all_coordinates_normalized.append(normalized_tmp)
            starting_y = starting_y - self.patch_size[1] // 2
        ret["go_posterior_gt_image"] = np.array(go_posterior_all_orig_imgs).astype(np.float32)
        ret["go_posterior_cond_image"] = np.array(go_posterior_all_cond_imgs).astype(np.float32)
        ret["go_posterior_mask"] = np.array(go_posterior_all_masks).astype(np.float32)
        ret["go_posterior_coordinates"] = np.array(go_posterior_all_coordinates)
        ret["go_posterior_coordinates_normalized"] = np.array(go_posterior_all_coordinates_normalized)

        # go anterior (y goes up) - the last 32 are padded, so we skip them
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_y = starting_y + self.patch_size[1] // 2
        go_anterior_all_orig_imgs = []
        go_anterior_all_cond_imgs = []
        go_anterior_all_masks = []
        go_anterior_all_coordinates = []
        go_anterior_all_coordinates_normalized = []
        while starting_y + self.patch_size[1] <= img.shape[1] - 32:
            patch_orig_img = img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            patch_blurry_cond_img = blurry_cond_img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            # use an anterior mask when going anterior
            patch_mask = self.get_mask(direction="anterior")
            patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                *patch_orig_img.shape
            )
            patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
            patch_all_cond_img = np.array(patch_all_cond_img)
            go_anterior_all_orig_imgs.append(patch_orig_img)
            go_anterior_all_cond_imgs.append(patch_all_cond_img)
            go_anterior_all_masks.append(patch_mask)
            tmp = np.array([starting_x, starting_y, starting_z])
            normalized_tmp = tmp * 2 / self.original_shape - 1
            go_anterior_all_coordinates.append(tmp)
            go_anterior_all_coordinates_normalized.append(normalized_tmp)
            starting_y = starting_y + self.patch_size[1] // 2
        ret["go_anterior_gt_image"] = np.array(go_anterior_all_orig_imgs).astype(np.float32)
        ret["go_anterior_cond_image"] = np.array(go_anterior_all_cond_imgs).astype(np.float32)
        ret["go_anterior_mask"] = np.array(go_anterior_all_masks).astype(np.float32)
        ret["go_anterior_coordinates"] = np.array(go_anterior_all_coordinates)
        ret["go_anterior_coordinates_normalized"] = np.array(go_anterior_all_coordinates_normalized)

        # after that, we use a batch of 5 patches to go inferior - z goes down
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_z = starting_z - self.patch_size[2] // 2
        go_inferior_all_orig_imgs = []
        go_inferior_all_cond_imgs = []
        go_inferior_all_masks = []
        go_inferior_all_coordinates = []
        go_inferior_all_coordinates_normalized = []
        while starting_z >= 0:
            go_inferior_column_orig_imgs = []
            go_inferior_column_cond_imgs = []
            go_inferior_column_masks = []
            go_inferior_column_coordinates = []
            go_inferior_column_coordinates_normalized = []
            for i in range(ydim_num_patches):
                starting_y = i * self.patch_size[1]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go inferior, we want to recover the inferior part of the image
                # thus use an inferior mask
                patch_mask = self.get_mask(direction="inferior")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)
                go_inferior_column_orig_imgs.append(patch_orig_img)
                go_inferior_column_cond_imgs.append(patch_all_cond_img)
                go_inferior_column_masks.append(patch_mask)
                tmp = np.array([starting_x, starting_y, starting_z])
                normalized_tmp = tmp * 2 / self.original_shape - 1
                go_inferior_column_coordinates.append(tmp)
                go_inferior_column_coordinates_normalized.append(normalized_tmp)
            # add the column to the list
            go_inferior_all_orig_imgs.append(go_inferior_column_orig_imgs)
            go_inferior_all_cond_imgs.append(go_inferior_column_cond_imgs)
            go_inferior_all_masks.append(go_inferior_column_masks)
            go_inferior_all_coordinates.append(go_inferior_column_coordinates)
            go_inferior_all_coordinates_normalized.append(go_inferior_column_coordinates_normalized)
            starting_z = starting_z - self.patch_size[2] // 2
        ret["go_inferior_gt_image"] = np.array(go_inferior_all_orig_imgs).astype(np.float32)
        ret["go_inferior_cond_image"] = np.array(go_inferior_all_cond_imgs).astype(np.float32)
        ret["go_inferior_mask"] = np.array(go_inferior_all_masks).astype(np.float32)
        ret["go_inferior_coordinates"] = np.array(go_inferior_all_coordinates)
        ret["go_inferior_coordinates_normalized"] = np.array(go_inferior_all_coordinates_normalized)

        # we also need to go superior (z goes up) - skip the last padded 32
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_z = starting_z + self.patch_size[2] // 2
        go_superior_all_orig_imgs = []
        go_superior_all_cond_imgs = []
        go_superior_all_masks = []
        go_superior_all_coordinates = []
        go_superior_all_coordinates_normalized = []
        while starting_z + self.patch_size[2] <= img.shape[2] - 32:
            go_superior_column_orig_imgs = []
            go_superior_column_cond_imgs = []
            go_superior_column_masks = []
            go_superior_column_coordinates = []
            go_superior_column_coordinates_normalized = []
            for i in range(ydim_num_patches):
                starting_y = i * self.patch_size[1]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go superior, we want to recover the superior part of the image
                # thus use a superior mask
                patch_mask = self.get_mask(direction="superior")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)
                go_superior_column_orig_imgs.append(patch_orig_img)
                go_superior_column_cond_imgs.append(patch_all_cond_img)
                go_superior_column_masks.append(patch_mask)
                tmp = np.array([starting_x, starting_y, starting_z])
                normalized_tmp = tmp * 2 / self.original_shape - 1
                go_superior_column_coordinates.append(tmp)
                go_superior_column_coordinates_normalized.append(normalized_tmp)
            # add the column to the list
            go_superior_all_orig_imgs.append(go_superior_column_orig_imgs)
            go_superior_all_cond_imgs.append(go_superior_column_cond_imgs)
            go_superior_all_masks.append(go_superior_column_masks)
            go_superior_all_coordinates.append(go_superior_column_coordinates)
            go_superior_all_coordinates_normalized.append(go_superior_column_coordinates_normalized)
            starting_z = starting_z + self.patch_size[2] // 2
        ret["go_superior_gt_image"] = np.array(go_superior_all_orig_imgs).astype(np.float32)
        ret["go_superior_cond_image"] = np.array(go_superior_all_cond_imgs).astype(np.float32)
        ret["go_superior_mask"] = np.array(go_superior_all_masks).astype(np.float32)
        ret["go_superior_coordinates"] = np.array(go_superior_all_coordinates)
        ret["go_superior_coordinates_normalized"] = np.array(go_superior_all_coordinates_normalized)

        # after that, we use a batch of ydim_num_patches * zdim_num_patches patches
        # to go left and right
        # x     0 left      x right
        # y     0 posterior y anterior
        # z     0 inferior  z superior
        # first we go left (x goes down)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_x = starting_x - self.patch_size[0] // 2
        go_left_all_orig_imgs = []
        go_left_all_cond_imgs = []
        go_left_all_masks = []
        go_left_all_coordinates = []
        go_left_all_coordinates_normalized = []

        while starting_x >= 0:
            go_left_slab_orig_imgs = []
            go_left_slab_cond_imgs = []
            go_left_slab_masks = []
            go_left_slab_coordinates = []
            go_left_slab_coordinates_normalized = []
            # for each y and z, we get a patch
            for i in range(ydim_num_patches * zdim_num_patches):
                nrow = i // zdim_num_patches  # each row has zdim_num_patches
                ncol = i % zdim_num_patches  # each column has ydim_num_patches
                starting_y = nrow * self.patch_size[1]
                starting_z = ncol * self.patch_size[2]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go left, we want to recover the left part of the image
                # thus use a left mask
                patch_mask = self.get_mask(direction="left")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)
                go_left_slab_orig_imgs.append(patch_orig_img)
                go_left_slab_cond_imgs.append(patch_all_cond_img)
                go_left_slab_masks.append(patch_mask)
                tmp = np.array([starting_x, starting_y, starting_z])
                normalized_tmp = tmp * 2 / self.original_shape - 1
                go_left_slab_coordinates.append(tmp)
                go_left_slab_coordinates_normalized.append(normalized_tmp)
            # add the slab to the list
            go_left_all_orig_imgs.append(go_left_slab_orig_imgs)
            go_left_all_cond_imgs.append(go_left_slab_cond_imgs)
            go_left_all_masks.append(go_left_slab_masks)
            go_left_all_coordinates.append(go_left_slab_coordinates)
            go_left_all_coordinates_normalized.append(go_left_slab_coordinates_normalized)
            starting_x = starting_x - self.patch_size[0] // 2
        ret["go_left_gt_image"] = np.array(go_left_all_orig_imgs).astype(np.float32)
        ret["go_left_cond_image"] = np.array(go_left_all_cond_imgs).astype(np.float32)
        ret["go_left_mask"] = np.array(go_left_all_masks).astype(np.float32)
        ret["go_left_coordinates"] = np.array(go_left_all_coordinates)
        ret["go_left_coordinates_normalized"] = np.array(go_left_all_coordinates_normalized)

        # now we go right (x goes up)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_x = starting_x + self.patch_size[0] // 2
        go_right_all_orig_imgs = []
        go_right_all_cond_imgs = []
        go_right_all_masks = []
        go_right_all_coordinates = []
        go_right_all_coordinates_normalized = []

        while starting_x + self.patch_size[0] <= img.shape[0]:
            go_right_slab_orig_imgs = []
            go_right_slab_cond_imgs = []
            go_right_slab_masks = []
            go_right_slab_coordinates = []
            go_right_slab_coordinates_normalized = []
            # for each y and z, we get a patch
            for i in range(ydim_num_patches * zdim_num_patches):
                nrow = i // zdim_num_patches
                ncol = i % zdim_num_patches
                starting_y = nrow * self.patch_size[1]
                starting_z = ncol * self.patch_size[2]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go right, we want to recover the right part of the image
                # thus use a right mask
                patch_mask = self.get_mask(direction="right")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)

                go_right_slab_orig_imgs.append(patch_orig_img)
                go_right_slab_cond_imgs.append(patch_all_cond_img)
                go_right_slab_masks.append(patch_mask)
                tmp = np.array([starting_x, starting_y, starting_z])
                normalized_tmp = tmp * 2 / self.original_shape - 1
                go_right_slab_coordinates.append(tmp)
                go_right_slab_coordinates_normalized.append(normalized_tmp)
            # add the slab to the list
            go_right_all_orig_imgs.append(go_right_slab_orig_imgs)
            go_right_all_cond_imgs.append(go_right_slab_cond_imgs)
            go_right_all_masks.append(go_right_slab_masks)
            go_right_all_coordinates.append(go_right_slab_coordinates)
            go_right_all_coordinates_normalized.append(go_right_slab_coordinates_normalized)
            starting_x = starting_x + self.patch_size[0] // 2
        ret["go_right_gt_image"] = np.array(go_right_all_orig_imgs).astype(np.float32)
        ret["go_right_cond_image"] = np.array(go_right_all_cond_imgs).astype(np.float32)
        ret["go_right_mask"] = np.array(go_right_all_masks).astype(np.float32)
        ret["go_right_coordinates"] = np.array(go_right_all_coordinates)
        ret["go_right_coordinates_normalized"] = np.array(go_right_all_coordinates_normalized)

        return ret


class TomsHCP3DTestPatchUncroppingGenerated(Dataset):
    def __init__(
        self,
        img_dir,
        patch_size=(64, 64, 64),
        full_mask_prob=0.0,
    ):
        print(f"Using image dir {img_dir}")
        self.img_dir = Path(img_dir)

        self.full_mask_prob = full_mask_prob
        self.mask_mode = "onedirection"

        # determine if patching is needed
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        self.patch_size = patch_size

        sample_img = nib.load(self.img_dir).get_fdata()
        if self.patch_size == sample_img.shape:
            self.patch = False
        else:
            self.patch = True

    def __len__(self):
        return 1

    def get_mask(self, direction="posterior"):
        if self.mask_mode == "manual":
            raise NotImplementedError("Manual mask generation is not implemented yet.")
            # mask = bbox2mask(self.patch_size, self.mask_config["shape"])
        elif self.mask_mode == "fourdirection" or self.mask_mode == "onedirection":
            mask = bbox2mask(
                self.patch_size,
                cropping_bbox_dir(
                    img_shape=self.patch_size,
                    mask_direction=direction,
                    random_direction=False,
                ),
            )
        elif self.mask_mode == "hybrid":
            raise NotImplementedError("Hybrid mask generation is not implemented yet.")
        elif self.mask_mode == "file":
            pass
        else:
            raise NotImplementedError(f"Mask mode {self.mask_mode} has not been implemented.")
        return mask.squeeze()

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        blurry_cond_img = nib.load(self.img_dir).get_fdata()
        img = np.zeros_like(blurry_cond_img) - 1.0

        # pad images to (224, 320, 256) (7 * 32, 10 * 32, 8 * 32)
        img = np.pad(img, ((0, 0), (16, 16), (16, 16)), mode="constant", constant_values=-1.0)
        assert img.shape == (224, 320, 256), f"img shape {img.shape} is not correct"
        blurry_cond_img = np.pad(blurry_cond_img, ((0, 0), (16, 16), (16, 16)), mode="constant", constant_values=-1.0)
        assert blurry_cond_img.shape == (224, 320, 256), f"blurry_cond_img shape {blurry_cond_img.shape} is not correct"

        # determine number of patches at each dimension
        xdim_num_patches = img.shape[0] // self.patch_size[0]
        ydim_num_patches = img.shape[1] // self.patch_size[1]
        zdim_num_patches = img.shape[2] // self.patch_size[2]
        print("xdim_num_patches", xdim_num_patches)
        print("ydim_num_patches", ydim_num_patches)
        print("zdim_num_patches", zdim_num_patches)

        # initial_patch in the center
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        initial_patch_orig_img = img[
            starting_x : starting_x + self.patch_size[0],
            starting_y : starting_y + self.patch_size[1],
            starting_z : starting_z + self.patch_size[2],
        ]
        initial_patch_blurry_cond_img = blurry_cond_img[
            starting_x : starting_x + self.patch_size[0],
            starting_y : starting_y + self.patch_size[1],
            starting_z : starting_z + self.patch_size[2],
        ]
        initial_mask = np.ones_like(initial_patch_orig_img)
        initial_masked_cond_img = initial_patch_orig_img * (1.0 - initial_mask) + initial_mask * np.random.randn(
            *initial_patch_orig_img.shape
        )
        initial_all_cond_img = [initial_patch_blurry_cond_img, initial_masked_cond_img]

        ret = {}
        ret["complete_gt_image"] = img
        ret["complete_cond_image"] = blurry_cond_img

        ret["initial_patch_gt_image"] = np.array(initial_patch_orig_img).astype(np.float32)
        ret["initial_patch_cond_image"] = np.array(initial_all_cond_img).astype(np.float32)
        ret["initial_patch_mask"] = np.array(initial_mask).astype(np.float32)
        ret["initial_patch_coordinates"] = np.array([starting_x, starting_y, starting_z])

        # x     0 left      x right
        # y     0 posterior y anterior
        # z     0 inferior  z superior
        # go posterior (y goes down)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_y = starting_y - self.patch_size[1] // 2
        go_posterior_all_orig_imgs = []
        go_posterior_all_cond_imgs = []
        go_posterior_all_masks = []
        go_posterior_all_coordinates = []
        while starting_y >= 0:
            patch_orig_img = img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            patch_blurry_cond_img = blurry_cond_img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            # use a posterior mask when going posterior
            patch_mask = self.get_mask(direction="posterior")
            patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                *patch_orig_img.shape
            )
            patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
            patch_all_cond_img = np.array(patch_all_cond_img)
            go_posterior_all_orig_imgs.append(patch_orig_img)
            go_posterior_all_cond_imgs.append(patch_all_cond_img)
            go_posterior_all_masks.append(patch_mask)
            go_posterior_all_coordinates.append([starting_x, starting_y, starting_z])
            starting_y = starting_y - self.patch_size[1] // 2
        ret["go_posterior_gt_image"] = np.array(go_posterior_all_orig_imgs).astype(np.float32)
        ret["go_posterior_cond_image"] = np.array(go_posterior_all_cond_imgs).astype(np.float32)
        ret["go_posterior_mask"] = np.array(go_posterior_all_masks).astype(np.float32)
        ret["go_posterior_coordinates"] = np.array(go_posterior_all_coordinates)

        # go anterior (y goes up)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_y = starting_y + self.patch_size[1] // 2
        go_anterior_all_orig_imgs = []
        go_anterior_all_cond_imgs = []
        go_anterior_all_masks = []
        go_anterior_all_coordinates = []
        while starting_y + self.patch_size[1] <= img.shape[1]:
            patch_orig_img = img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            patch_blurry_cond_img = blurry_cond_img[
                starting_x : starting_x + self.patch_size[0],
                starting_y : starting_y + self.patch_size[1],
                starting_z : starting_z + self.patch_size[2],
            ]
            # use an anterior mask when going anterior
            patch_mask = self.get_mask(direction="anterior")
            patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                *patch_orig_img.shape
            )
            patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
            patch_all_cond_img = np.array(patch_all_cond_img)
            go_anterior_all_orig_imgs.append(patch_orig_img)
            go_anterior_all_cond_imgs.append(patch_all_cond_img)
            go_anterior_all_masks.append(patch_mask)
            go_anterior_all_coordinates.append([starting_x, starting_y, starting_z])
            starting_y = starting_y + self.patch_size[1] // 2
        ret["go_anterior_gt_image"] = np.array(go_anterior_all_orig_imgs).astype(np.float32)
        ret["go_anterior_cond_image"] = np.array(go_anterior_all_cond_imgs).astype(np.float32)
        ret["go_anterior_mask"] = np.array(go_anterior_all_masks).astype(np.float32)
        ret["go_anterior_coordinates"] = np.array(go_anterior_all_coordinates)

        # after that, we use a batch of 5 patches to go inferior - z goes down
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_z = starting_z - self.patch_size[2] // 2
        go_inferior_all_orig_imgs = []
        go_inferior_all_cond_imgs = []
        go_inferior_all_masks = []
        go_inferior_all_coordinates = []
        while starting_z >= 0:
            go_inferior_column_orig_imgs = []
            go_inferior_column_cond_imgs = []
            go_inferior_column_masks = []
            go_inferior_column_coordinates = []
            for i in range(ydim_num_patches):
                starting_y = i * self.patch_size[1]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go inferior, we want to recover the inferior part of the image
                # thus use an inferior mask
                patch_mask = self.get_mask(direction="inferior")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)
                go_inferior_column_orig_imgs.append(patch_orig_img)
                go_inferior_column_cond_imgs.append(patch_all_cond_img)
                go_inferior_column_masks.append(patch_mask)
                go_inferior_column_coordinates.append([starting_x, starting_y, starting_z])
            # add the column to the list
            go_inferior_all_orig_imgs.append(go_inferior_column_orig_imgs)
            go_inferior_all_cond_imgs.append(go_inferior_column_cond_imgs)
            go_inferior_all_masks.append(go_inferior_column_masks)
            go_inferior_all_coordinates.append(go_inferior_column_coordinates)
            starting_z = starting_z - self.patch_size[2] // 2
        ret["go_inferior_gt_image"] = np.array(go_inferior_all_orig_imgs).astype(np.float32)
        ret["go_inferior_cond_image"] = np.array(go_inferior_all_cond_imgs).astype(np.float32)
        ret["go_inferior_mask"] = np.array(go_inferior_all_masks).astype(np.float32)
        ret["go_inferior_coordinates"] = np.array(go_inferior_all_coordinates)

        # we also need to go superior (z goes up)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_z = starting_z + self.patch_size[2] // 2
        go_superior_all_orig_imgs = []
        go_superior_all_cond_imgs = []
        go_superior_all_masks = []
        go_superior_all_coordinates = []
        while starting_z + self.patch_size[2] <= img.shape[2]:
            go_superior_column_orig_imgs = []
            go_superior_column_cond_imgs = []
            go_superior_column_masks = []
            go_superior_column_coordinates = []
            for i in range(ydim_num_patches):
                starting_y = i * self.patch_size[1]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go superior, we want to recover the superior part of the image
                # thus use a superior mask
                patch_mask = self.get_mask(direction="superior")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)
                go_superior_column_orig_imgs.append(patch_orig_img)
                go_superior_column_cond_imgs.append(patch_all_cond_img)
                go_superior_column_masks.append(patch_mask)
                go_superior_column_coordinates.append([starting_x, starting_y, starting_z])
            # add the column to the list
            go_superior_all_orig_imgs.append(go_superior_column_orig_imgs)
            go_superior_all_cond_imgs.append(go_superior_column_cond_imgs)
            go_superior_all_masks.append(go_superior_column_masks)
            go_superior_all_coordinates.append(go_superior_column_coordinates)
            starting_z = starting_z + self.patch_size[2] // 2
        ret["go_superior_gt_image"] = np.array(go_superior_all_orig_imgs).astype(np.float32)
        ret["go_superior_cond_image"] = np.array(go_superior_all_cond_imgs).astype(np.float32)
        ret["go_superior_mask"] = np.array(go_superior_all_masks).astype(np.float32)
        ret["go_superior_coordinates"] = np.array(go_superior_all_coordinates)

        # after that, we use a batch of ydim_num_patches * zdim_num_patches patches
        # to go left and right
        # x     0 left      x right
        # y     0 posterior y anterior
        # z     0 inferior  z superior
        # first we go left (x goes down)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_x = starting_x - self.patch_size[0] // 2
        go_left_all_orig_imgs = []
        go_left_all_cond_imgs = []
        go_left_all_masks = []
        go_left_all_coordinates = []

        while starting_x >= 0:
            go_left_slab_orig_imgs = []
            go_left_slab_cond_imgs = []
            go_left_slab_masks = []
            go_left_slab_coordinates = []
            # for each y and z, we get a patch
            for i in range(ydim_num_patches * zdim_num_patches):
                nrow = i // zdim_num_patches  # each row has zdim_num_patches
                ncol = i % zdim_num_patches  # each column has ydim_num_patches
                starting_y = nrow * self.patch_size[1]
                starting_z = ncol * self.patch_size[2]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go left, we want to recover the left part of the image
                # thus use a left mask
                patch_mask = self.get_mask(direction="left")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)
                go_left_slab_orig_imgs.append(patch_orig_img)
                go_left_slab_cond_imgs.append(patch_all_cond_img)
                go_left_slab_masks.append(patch_mask)
                go_left_slab_coordinates.append([starting_x, starting_y, starting_z])
            # add the slab to the list
            go_left_all_orig_imgs.append(go_left_slab_orig_imgs)
            go_left_all_cond_imgs.append(go_left_slab_cond_imgs)
            go_left_all_masks.append(go_left_slab_masks)
            go_left_all_coordinates.append(go_left_slab_coordinates)
            starting_x = starting_x - self.patch_size[0] // 2
        ret["go_left_gt_image"] = np.array(go_left_all_orig_imgs).astype(np.float32)
        ret["go_left_cond_image"] = np.array(go_left_all_cond_imgs).astype(np.float32)
        ret["go_left_mask"] = np.array(go_left_all_masks).astype(np.float32)
        ret["go_left_coordinates"] = np.array(go_left_all_coordinates)

        # now we go right (x goes up)
        starting_x = 32 * 3
        starting_y = 32 * 4
        starting_z = 32 * 3
        starting_x = starting_x + self.patch_size[0] // 2
        go_right_all_orig_imgs = []
        go_right_all_cond_imgs = []
        go_right_all_masks = []
        go_right_all_coordinates = []

        while starting_x + self.patch_size[0] <= img.shape[0]:
            go_right_slab_orig_imgs = []
            go_right_slab_cond_imgs = []
            go_right_slab_masks = []
            go_right_slab_coordinates = []
            # for each y and z, we get a patch
            for i in range(ydim_num_patches * zdim_num_patches):
                nrow = i // zdim_num_patches
                ncol = i % zdim_num_patches
                starting_y = nrow * self.patch_size[1]
                starting_z = ncol * self.patch_size[2]
                patch_orig_img = img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                patch_blurry_cond_img = blurry_cond_img[
                    starting_x : starting_x + self.patch_size[0],
                    starting_y : starting_y + self.patch_size[1],
                    starting_z : starting_z + self.patch_size[2],
                ]
                # when we go right, we want to recover the right part of the image
                # thus use a right mask
                patch_mask = self.get_mask(direction="right")
                patch_masked_cond_img = patch_orig_img * (1.0 - patch_mask) + patch_mask * np.random.randn(
                    *patch_orig_img.shape
                )
                patch_all_cond_img = [patch_blurry_cond_img, patch_masked_cond_img]
                patch_all_cond_img = np.array(patch_all_cond_img)

                go_right_slab_orig_imgs.append(patch_orig_img)
                go_right_slab_cond_imgs.append(patch_all_cond_img)
                go_right_slab_masks.append(patch_mask)
                go_right_slab_coordinates.append([starting_x, starting_y, starting_z])
            # add the slab to the list
            go_right_all_orig_imgs.append(go_right_slab_orig_imgs)
            go_right_all_cond_imgs.append(go_right_slab_cond_imgs)
            go_right_all_masks.append(go_right_slab_masks)
            go_right_all_coordinates.append(go_right_slab_coordinates)
            starting_x = starting_x + self.patch_size[0] // 2
        ret["go_right_gt_image"] = np.array(go_right_all_orig_imgs).astype(np.float32)
        ret["go_right_cond_image"] = np.array(go_right_all_cond_imgs).astype(np.float32)
        ret["go_right_mask"] = np.array(go_right_all_masks).astype(np.float32)
        ret["go_right_coordinates"] = np.array(go_right_all_coordinates)

        return ret
