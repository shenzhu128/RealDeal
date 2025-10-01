<h1 align="center">
  <b>RealDeal</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.11-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-2.5-2BAF2B.svg" /></a>
      <a href= "https://arxiv.org/abs/2507.18830">
        <img src="https://img.shields.io/badge/arXiv-2507.18830-b31b1b.svg" /></a>
</p>

**Official implementation for paper:**

[**RealDeal: Enhancing Realism and Details in Brain Image Generation via Image-to-Image Diffusion Models**](https://arxiv.org/abs/2507.18830)<br/>
[Shen Zhu](https://www.linkedin.com/in/shen-z-7659471a4/), 
[Yinzhu Jin](https://www.linkedin.com/in/yinzhu-jin-635514227/), 
Tyler Spears, 
Ifrah Zawar,
P. Thomas Fletcher 

<p align="center">
  <img src=assets/qualitative_recon.png />
  <img src=assets/model.png />
</p>



---

### BibTeX

To cite our work, you can use:
```
@inproceedings{zhu2025realdeal,
  title={RealDeal: Enhancing Real ism and De tails in Brain Image Generation via Image-to-Image Diffusion Models},
  author={Zhu, Shen and Jin, Yinzhu and Spears, Tyler and Zawar, Ifrah and Fletcher, P Thomas},
  booktitle={MICCAI Workshop on Deep Generative Models},
  pages={151--160},
  year={2025},
  organization={Springer}
}
```

---

### News
**Oct. 1, 2025:** Added the required file `fractions_64.npy` in Google Drive.

**Sept. 30, 2025:** Added script for autoencoder, latent diffusion, and patch-based diffusion.

---

### Requirements
- Python >= 3.11
- PyTorch >= 2.5
- CUDA enabled computing device
- Download `fractions_64.npy` at [this link](https://drive.google.com/drive/folders/1opL9r4nqrAh4fqMVL3Qr_UajNoX8-lJI?usp=sharing) and put it in the data directory
- Download checkpoints at [this link](https://drive.google.com/drive/folders/1opL9r4nqrAh4fqMVL3Qr_UajNoX8-lJI?usp=sharing)

### Installation
```
$ git clone https://github.com/shenzhu128/RealDeal.git
$ cd RealDeal
$ conda env create -f env.yaml
$ conda activate realdeal
```

### To train the autoencoder

For the training data, we assume the following data folder structure:

```
.
├── subject1/
│   └── T1w/
│       └── T1w_cropped.nii.gz
├── subject2/
│   └── T1w/
│       └── T1w_cropped.nii.gz
├── ...
├── fractions_64.npy
└── hcp_split.csv
```

The input `T1w_cropped.nii.gz` is of shape `[224, 288, 224]` and scaled to the range of `[-1, 1]`. We downsize the input 4x, and the latent size for the autoencoder is `[4, 56, 72, 56]` with 4 channels.

The commands for train, encode inputs, and decode latent codes are in `autoencoder/run.sh`. Please adjust the parameters accordingly. A checkpoint is provided for the autoencoder `autoencoder/autoencoder_checkpoint.pth`.

After encoding the inputs with `EncodeHCP.py` and the provided checkpoint, your folder structure should look like:
```
.
├── subject1/
│   └── T1w/
│       ├── T1w_cropped.nii.gz
│       └── T1w_encoded.npy
├── subject2/
│   └── T1w/
│       ├── T1w_cropped.nii.gz
│       └── T1w_encoded.npy
├── ...
├── fractions_64.npy
└── hcp_split.csv
```

### To train the latent diffusion model

With the encodings from previous step, we can train the latent diffusion model for generating synthetic brain MRIs. However, one important step is to scale the encodings channel-wise, so the data ranges roughly from -1 to 1. This way the latent diffusion model learns better.

You need to first find the data range for each channel of all your encodings. And then adjust the  `cutoff_vals` in `autoencoder/scale_andunscale.ipynb` according to your own encodings. Next, use function `scale_img()` for all your encodings. Name each scaled encoding as `T1w_encoded_rescaled.npy`. And your final folder structure should look like this:
```
.
├── subject1/
│   └── T1w/
│       ├── T1w_cropped.nii.gz
│       ├── T1w_encoded.npy
│       └── T1w_encoded_rescaled.npy
├── subject2/
│   └── T1w/
│       ├── T1w_cropped.nii.gz
│       ├── T1w_encoded.npy
│       └── T1w_encoded_rescaled.npy
├── ...
├── fractions_64.npy
└── hcp_split.csv
```

With the training data `T1w_encoded_rescaled.npy`, use the following script to train the latent diffusion model:
```
$ cd latent_diffusion_patch_diffusion/src/bash
$ bash ldm.sh
```

### To sample from the latent diffusion model

Use the following script to sample from the latent diffusion model. The output is an encoding that has the same range as `T1w_encoded_rescaled.npy`. Thus, after sampling, you need to unscale it using the `unscale_img()` function in `autoencoder/scale_andunscale.ipynb`. And then decode the sampled-unscaled encoding using the autoencoder (script `autoencoder/DecodeHCP.py`) to get a synthetic brain image. 
```
cd latent_diffusion_patch_diffusion/src/python/sampling
bash sampling.sh
```

Change the `diffusion_root_path`, `ckpt`, and `need_to_sample` parameters to your situation.

You can use the provided checkpoint to sample from the latent space. Checkpoint name is `checkpoints/checkpoint750.pth`.

### To train the patch-based diffusion model - RealDeal

To train the patch-based diffusion model, we need the reconstructed training images using the trained autoencoder. Use script `autoencoder/DecodeHCP.py`, and the final folder structure should look like this:
```
.
├── subject1/
│   └── T1w/
│       ├── T1w_cropped.nii.gz
│       ├── T1w_encoded.npy
│       ├── T1w_encoded_rescaled.npy
│       └── T1w_reconstructed.nii.gz
├── subject2/
│   └── T1w/
│       ├── T1w_cropped.nii.gz
│       ├── T1w_encoded.npy
│       ├── T1w_encoded_rescaled.npy
│       └── T1w_reconstructed.nii.gz
├── ...
├── fractions_64.npy
└── hcp_split.csv
```

Then, run the 
```
$ cd latent_diffusion_patch_diffusion/src/bash
$ bash patch_diffusion_hcp3d.sh
```

### To apply RealDeal to images

For reconstructed test images, or generated synthetic images, run the following commands to apply RealDeal:

```
# For reconstructed images
cd latent_diffusion_patch_diffusion/src/python/testing
bash recover_hcp3d_with_modified_group_norm.sh
```

```
# For generated synthetic images
cd latent_diffusion_patch_diffusion/src/python/testing
bash recover_hcp3d_generated.sh
```

Adjust the parameters in the script to change input data or checkpoints. Checkpoint for trained RealDeal model is available. Checkpoint name is `checkpoints/checkpoint2690.pth`.

