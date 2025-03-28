#!/bin/bash

# YOU MUST SET THE CUDA_HOME AND PATH AND LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=5

echo "start time: $(date)"

python testing/evaluation.py \
    --data_root '.' \
    --model_path 'THUDM/CogVideoX-5b-I2V' \
    --evaluation_dir 'eval_res/250218_depth_seen' \
    --fps 8 \
    --num_samples 3 \
    --generate_type i2v \
    --tracking_column data/dexycb_filelist/dexycb_depth_training_depths.txt \
    --video_column data/dexycb_filelist/dexycb_depth_training_videos.txt \
    --caption_column data/dexycb_filelist/dexycb_depth_training_prompts.txt \
    --transformer_path exps/cogshader_HOI_depth_100000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-2000-convert
    # --image_paths repaint.txt \