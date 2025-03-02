import os
import shutil

from tqdm import tqdm

def split_path_func(path, split_list=["0", "1", "2", "3"]):
    split_path = path.split("/")
    for i, split in enumerate(split_path):
        if split in split_list:
            split_path[i] = ""
    
    # Remove empty strings
    split_path = list(filter(lambda x: x != "", split_path))
    return "/".join(split_path)
    

def merge_latent_files(
    root_dir: str,
    output_dir: str,
    ext_lits: list = ["png", "mp4", "pt"],
):
    for root, dirs, files in tqdm(os.walk(root_dir)):
        for file in files:
            if file.split(".")[-1] in ext_lits:
                file_path = os.path.join(root, file)
                new_path = file_path.replace(root_dir, output_dir)
                new_path = split_path_func(new_path)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copyfile(file_path, new_path)

def count_files(dir_path: str):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        count += len(files)
    return count
                
if __name__ == "__main__":
    root_dir = 'data/dexycb_latents'
    # output_dir = 'data/dexycb_latents'
    # merge_latent_files(root_dir, output_dir)
    
    for dir_path in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_path)
        print(f"{dir_path}: {count_files(dir_path)}")
    