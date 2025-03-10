import random
import os

MAX_NUM = 2999

random_indices = random.sample(range(MAX_NUM), 500)

ROOT_DIR = '../data/dexycb_filelist/training_latents'
OUTPUT_DIR = '../data/dexycb_filelist/training_latents_partial'

for filelist in os.listdir(ROOT_DIR):
    with open(os.path.join(ROOT_DIR, filelist), 'r') as f:
        lines = f.readlines()
        with open(os.path.join(OUTPUT_DIR, filelist), 'w') as out:
            for i in random_indices:
                out.write(lines[i])
