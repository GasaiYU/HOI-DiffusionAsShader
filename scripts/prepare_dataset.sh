export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 training/prepare_dataset_hoi.py \
    --data_root "." \
    --caption_column "data/dexycb_filelist/training/training_prompts.txt" \
    --video_column "data/dexycb_filelist/training/training_videos.txt" \
    --tracking_column "data/dexycb_filelist/training/training_trackings.txt" \
    --normal_column "data/dexycb_filelist/training/training_normals.txt" \
    --depth_column "data/dexycb_filelist/training/training_depths.txt" \
    --label_column "data/dexycb_filelist/training/training_labels.txt" \
    --height_buckets 480 --width_buckets 720 \
    --save_image_latents --output_dir "data/dexycb_latents_new" \
    --target_fps 15 --save_latents_and_embeddings

