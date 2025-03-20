export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 training/prepare_dataset_hoi.py \
    --data_root "." \
    --caption_column "data/dexycb_filelist/training_partial/training_prompts.txt" \
    --video_column "data/dexycb_filelist/training_partial/training_videos.txt" \
    --tracking_column "data/dexycb_filelist/training_partial/training_trackings.txt" \
    --normal_column "data/dexycb_filelist/training_partial/training_normals.txt" \
    --depth_column "data/dexycb_filelist/training_partial/training_depths.txt" \
    --label_column "data/dexycb_filelist/training_partial/training_labels.txt" \
    --height_buckets 480 --width_buckets 720 \
    --save_image_latents --output_dir "data/dexycb_latents_partial_fixed_tmp" \
    --target_fps 15 --save_latents_and_embeddings

