/scratch/sz9jt/.conda/envs/generativebrain/bin/python /home/sz9jt/projects/generative_brain/src/python/encode-decode-sample/reconstruct.py \
--device="cuda:0" \
--dataset_type="brain" \
--config_path="/home/sz9jt/data/generative_brain/stage1/aekl_corrected_v0_2x_3ch/config.yaml" \
--weight="/home/sz9jt/data/generative_brain/stage1/aekl_corrected_v0_2x_3ch/checkpoint104.pth" \
--outpath="/home/sz9jt/data/generative_brain/stage1/aekl_corrected_v0_2x_3ch/decoded_test_ds_ckpt104" \
--corrected="True" \
--resample="True"