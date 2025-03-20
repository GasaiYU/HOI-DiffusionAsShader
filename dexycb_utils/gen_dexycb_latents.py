import os
import shutil

def judge_file_exist(root_dir, keys, file):
    for data_key in keys:
        if file.endswith(".mp4") and not data_key == "videos":
            if not os.path.exists(os.path.join(root_dir, data_key, file.replace(".mp4", ".pt"))):
                return False
        elif data_key == "videos":
            if not os.path.exists(os.path.join(root_dir, data_key, file.replace(".pt", ".mp4"))):
                return False
        elif not os.path.exists(os.path.join(root_dir, data_key, file)):
            return False
    return True

def gen_dexycb_latents(data_key, root_dir, save_dir, keys):
    files = list(os.listdir(os.path.join(root_dir, data_key)))
    files.sort(key=lambda x: int(x.split('.')[0].replace('-', ''), base=16))
    
    with open(os.path.join(save_dir, f"training_{data_key}.txt"), 'w') as f:
        for file in files:
            if judge_file_exist(root_dir, keys, file):
                f.write(f"{os.path.join(root_dir, data_key, file)}\n")

    with open(os.path.join(save_dir, f"training_{data_key}.txt"), 'r') as f:
        lines = f.readlines()
    print(f"training_{data_key}.txt: {len(lines)}")

if __name__ == "__main__":
    root_dir = "data/dexycb_latents_partial_fixed"
    save_dir = "data/dexycb_filelist/training_latents_partial_new"
    os.makedirs(save_dir, exist_ok=True)
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
        gen_dexycb_latents(key, root_dir, save_dir, keys)
    