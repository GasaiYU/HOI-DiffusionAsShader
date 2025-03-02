PROMPT='A hand is holding a wine bottle'
OUTPUT_DIR="eval_res/test_simulation/20001"
INPUT_PATH="val_resources/simulation/fourth20001-0100.mp4"
CHECKPOINT_PATH="THUDM/CogVideoX-5b-I2V"
TRANSFORMER_PATH='exps/cogshader_HOI_100000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-2000-convert'
GPU=5

cmd="python demo.py \
    --prompt \"$PROMPT\" --checkpoint_path $CHECKPOINT_PATH --output_dir $OUTPUT_DIR \
    --input_path $INPUT_PATH  --gpu $GPU --transformer_path $TRANSFORMER_PATH"

echo "Running command: $cmd"
eval $cmd