import os
import shutil

def gen_dexycb_latents(data_key, root_dir, save_dir):
    files = list(os.listdir(os.path.join(root_dir, data_key)))
    files.sort(key=lambda x: int(x.split('.')[0].replace('-', ''), base=16))
    
    with open(os.path.join(save_dir, f"training_{data_key}.txt"), 'w') as f:
        for file in files:
            f.write(f"{os.path.join(root_dir, data_key, file)}\n")

if __name__ == "__main__":
    root_dir = "data/dexycb_latents"
    save_dir = "data/dexycb_filelist/training_latents"
    
    keys = ["depth_images_latents", 
            "depth_latents", 
            "hand_keypoints_latents",
            "hand_keypoints_images_latents",
            "image_latents",
            "normal_images_latents",
            "normal_latents",
            "prompt_embeds",
            "seg_mask_images_latents",
            "seg_mask_latents",
            "tracking_images_latents",
            "tracking_latents",
            "video_latents",
            "videos"]
    
    for key in keys:
        gen_dexycb_latents(key, root_dir, save_dir)