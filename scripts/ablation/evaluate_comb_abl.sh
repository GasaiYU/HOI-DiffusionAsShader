#!/bin/bash

# YOU MUST SET THE CUDA_HOME AND PATH AND LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=5

echo "start time: $(date)"

python testing/evaluation_comb.py \
    --data_root '.' \
    --model_path 'THUDM/CogVideoX-5b-I2V' \
    --evaluation_dir 'eval_res/comb_abl' \
    --fps 15 \
    --num_samples 1 \
    --generate_type i2v \
    --tracking_column data/dexycb_filelist/val_trackings.txt \
    --video_column data/dexycb_filelist/val_videos.txt \
    --caption_column data/dexycb_filelist/val_prompts.txt \
    --depth_column data/dexycb_filelist/val_depths.txt \
    --label_column data/dexycb_filelist/val_labels.txt \
    --normal_column data/dexycb_filelist/val_depths.txt \
    --transformer_path exps/cogshader_HOI_comb_latents_no_depth_100000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-1300-convert
    # --image_paths repaint.txt \