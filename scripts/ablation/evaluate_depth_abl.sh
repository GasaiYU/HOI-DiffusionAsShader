#!/bin/bash

# YOU MUST SET THE CUDA_HOME AND PATH AND LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=5

echo "start time: $(date)"

python testing/evaluation.py \
    --data_root '.' \
    --model_path 'THUDM/CogVideoX-5b-I2V' \
    --evaluation_dir 'eval_res/depth_abl' \
    --fps 15 \
    --num_samples 200 \
    --generate_type i2v \
    --tracking_column data/dexycb_filelist/val_depths.txt \
    --video_column data/dexycb_filelist/val_videos.txt \
    --caption_column data/dexycb_filelist/val_prompts.txt \
    --transformer_path exps/cogshader_HOI_depth_100000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-2000-convert
    # --image_paths repaint.txt \