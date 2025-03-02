python demo_dexycb.py \
    --input_filelist 'data/val_trackings_seen.txt' \
    --prompt_filelist 'data/val_prompts_seen.txt' \
    --output_dir './val_res/val_seen_0218' \
    --checkpoint_path 'THUDM/CogVideoX-5b-I2V' \
    --gpu 0 \
    --tracking_filelist 'data/val_trackings_seen.txt' \
    --transformer_path 'exps/cogshader_HOI_100000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-2000-convert' \
