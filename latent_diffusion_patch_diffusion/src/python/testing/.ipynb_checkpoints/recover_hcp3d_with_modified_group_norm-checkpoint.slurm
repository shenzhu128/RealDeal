#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --account=manifold
#SBATCH -o ./recover_hcp3d_with_modifed_group_norm.out

conda activate genbrain
cd "/home/sz9jt/projects/generative_brain/src/python/testing"
python recover_hcp3d_with_modified_group_norm.py \
--ckpt_path "/home/sz9jt/manifold/sz9jt/generative_brain/cond_diffusion/diffusion_tomshcp3d_maskedcond_finetune_modify_group_norm" \
--modified_group_norm_ckpt 2690