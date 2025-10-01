# Adjust the diffusion_root_path and ckpt as needed
# need_to_sample indicates the index of the output samples you want to generate
# Run multiple times with different need_to_sample values to parallelize sampling
/home/sz9jt/.conda/envs/mrinr/bin/python sampling.py \
--diffusion_root_path="/home/sz9jt/manifold/sz9jt/realdeal/diffusion/diffusion_tomhcp3dlatent_4ch" \
--ckpt="750" \
--device_rank=0 \
--samples_per_gpu=100 \
--need_to_sample 4 5 6