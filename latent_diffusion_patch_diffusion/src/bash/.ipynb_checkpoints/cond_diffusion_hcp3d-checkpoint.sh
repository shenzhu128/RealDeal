# python /home/sz9jt/projects/generative_brain/src/python/training/train_cond_diffusion.py \
# --seed=42 \
# --output_dir="/home/sz9jt/data/generative_brain/cond_diffusion" \
# --run_dir="diffusion_hcp3d_cond" \
# --config_file="/home/sz9jt/projects/generative_brain/configs/cond_diff/diffusion_hcp3d_cond.yaml" \
# --batch_size=20 \
# --num_workers=4 \
# --project="cond-diffusion" \
# --n_epochs=100 \
# --eval_freq=1

# python /home/sz9jt/projects/generative_brain/src/python/training/train_cond_diffusion.py \
# --seed=42 \
# --output_dir="/home/sz9jt/data/generative_brain/cond_diffusion" \
# --run_dir="diffusion_tomshcp3d_cond" \
# --config_file="/home/sz9jt/projects/generative_brain/configs/cond_diff/diffusion_tomshcp3d_cond.yaml" \
# --batch_size=20 \
# --num_workers=4 \
# --project="cond-diffusion" \
# --n_epochs=300 \
# --eval_freq=10 \
# --enable_wandb=0

# python /home/sz9jt/projects/generative_brain/src/python/training/train_cond_diffusion.py \
# --seed=42 \
# --output_dir="/home/sz9jt/data/generative_brain/cond_diffusion" \
# --run_dir="diffusion_tomshcp3d_maskedcond" \
# --config_file="/home/sz9jt/projects/generative_brain/configs/cond_diff/diffusion_tomshcp3d_maskedcond.yaml" \
# --batch_size=32 \
# --num_workers=4 \
# --project="cond-diffusion" \
# --n_epochs=2000 \
# --eval_freq=10 \
# --enable_wandb=0 \
# --task="uncropping"

# python /home/sz9jt/projects/generative_brain/src/python/training/train_cond_diffusion.py \
# --seed=42 \
# --output_dir="/home/sz9jt/manifold/sz9jt/generative_brain/cond_diffusion" \
# --run_dir="diffusion_tomshcp3d_maskedcond_finetune" \
# --config_file="/home/sz9jt/projects/generative_brain/configs/cond_diff/diffusion_tomshcp3d_maskedcond_finetune.yaml" \
# --batch_size=32 \
# --num_workers=4 \
# --project="cond-diffusion" \
# --n_epochs=2000 \
# --eval_freq=10 \
# --enable_wandb=0 \
# --task="uncropping"

# python /home/sz9jt/projects/generative_brain/src/python/training/train_cond_diffusion_with_coordinates.py \
# --seed=42 \
# --output_dir="/home/sz9jt/manifold/sz9jt/generative_brain/cond_diffusion" \
# --run_dir="diffusion_tomshcp3d_maskedcond_with_coordinates" \
# --config_file="/home/sz9jt/projects/generative_brain/configs/cond_diff/diffusion_tomshcp3d_maskedcond_with_coordinates.yaml" \
# --batch_size=32 \
# --num_workers=4 \
# --project="cond-diffusion" \
# --n_epochs=2000 \
# --eval_freq=10 \
# --enable_wandb=0 \
# --task="uncropping"

python /home/sz9jt/projects/generative_brain/src/python/training/train_cond_diffusion_modify_group_norm.py \
--seed=42 \
--output_dir="/home/sz9jt/manifold/sz9jt/generative_brain/cond_diffusion" \
--run_dir="diffusion_tomshcp3d_maskedcond_finetune_modify_group_norm" \
--config_file="/home/sz9jt/projects/generative_brain/configs/cond_diff/diffusion_tomshcp3d_maskedcond_finetune_modify_group_norm.yaml" \
--batch_size=32 \
--num_workers=4 \
--project="cond-diffusion" \
--n_epochs=3000 \
--eval_freq=10 \
--enable_wandb=0 \
--task="uncropping"