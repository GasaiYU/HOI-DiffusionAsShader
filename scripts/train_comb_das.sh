export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0"
PORT=29505

LEARNING_RATES=("1e-4")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("100000")

# training dataset parameters
DATA_ROOT="."
MODEL_PATH="THUDM/CogVideoX-5b-I2V"
CAPTION_COLUMN="data/dexycb_filelist/training/training_prompts.txt"
VIDEO_COLUMN="data/dexycb_filelist/training/training_videos.txt"
TRACKING_COLUMN="data/dexycb_filelist/training/training_trackings.txt"
DEPTH_COLUMN="data/dexycb_filelist/training/training_depths.txt"
NORMAL_COLUMN="data/dexycb_filelist/training/training_normals.txt"
LABLE_COLUMN="data/dexycb_filelist/training/training_labels.txt"


# validation parameters
TRACKING_MAP_PATH="./val_resources/comb_val/val_tracking.mp4"
DEPTH_MAP_PATH="./val_resources/comb_val/val_depth.mp4"
NORMAL_MAP_PATH="./val_resources/comb_val/val_normal.mp4"
SEG_MASK_PATH="./val_resources/comb_val/val_seg_mask.mp4"
HAND_KEYPOINTS_PATH="./val_resources/comb_val/val_hand_keypoints.mp4"
VALIDATION_PROMPT="Initially, a man in a dark t-shirt and blue jeans is seen in an industrial setting, holding a red object with a focused expression, surrounded by a black table with a white box and a red mug, and a robotic arm with a camera. The scene shifts to the man in a black t-shirt and blue jeans, holding a red object with a focused expression, with a robotic arm and a camera system nearby, suggesting a demonstration or testing scenario. Finally, the man, now in a black t-shirt and blue shorts, holds a red cube with a focused expression, in an environment equipped with a robotic arm and a camera system, indicating a technological demonstration or testing scenario."
VALIDATION_IMAGES="./val_resources/comb_val/val_image.jpg"

ACCELERATE_CONFIG_FILE="./accelerate_configs/deepspeed.yaml"

TRAIN_BATCH_SIZE=1
CHECKPOINT_STEPS=500
WARMUP_STEPS=500

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="./exps/cogshader_HOI_comb_test_${steps}__optimizer_${optimizer}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS --main_process_port $PORT training/cogvideox_image_to_video_comb_sft.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --tracking_column $TRACKING_COLUMN \
          --tracking_map_path $TRACKING_MAP_PATH \
          --depth_column $DEPTH_COLUMN \
          --depth_map_path $DEPTH_MAP_PATH \
          --normal_column $NORMAL_COLUMN \
          --normal_map_path $NORMAL_MAP_PATH \
          --label_column $LABLE_COLUMN \
          --seg_mask_path $SEG_MASK_PATH \
          --hand_keypoints_path $HAND_KEYPOINTS_PATH \
          --num_tracking_blocks 18 \
          --height_buckets 480 \
          --width_buckets 720 \
          --height 480 \
          --width 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 4 \
          --pin_memory \
          --validation_prompt \"$VALIDATION_PROMPT\" \
          --validation_images $VALIDATION_IMAGES \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size $TRAIN_BATCH_SIZE \
          --max_train_steps $steps \
          --checkpointing_steps $CHECKPOINT_STEPS \
          --gradient_accumulation_steps 4 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps $WARMUP_STEPS \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --noised_image_dropout 0.05 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --resume_from_checkpoint \"latest\" \
          --nccl_timeout 1800 --checkpoints_total_limit 5 --initial_frames_num 1
          "
    
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
