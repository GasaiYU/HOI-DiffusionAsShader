import os
import shutil 

if __name__ == "__main__":
    os.chdir('..')

    depth_dir = 'data/dexycb_depth'
    merged_depth_dir = 'data/dexycb_inferred_depth'

    existing_depth_paths = []
    for root, _, files in os.walk(depth_dir):
        for file in files:
            if file.endswith('masked_color.mp4'):
                existing_depth_paths.append(os.path.join(root, file))

    for root, _, files in os.walk(merged_depth_dir):
        for file in files:
            if file.endswith('masked_color.mp4') or file.endswith('depth.mp4'):
                merged_file_path = os.path.join(root, file).replace('dexycb_inferred_depth', 'dexycb_depth')
                if merged_file_path not in existing_depth_paths:
                    if not os.path.exists(os.path.dirname(merged_file_path)):
                        os.makedirs(os.path.dirname(merged_file_path), exist_ok=True)
                    shutil.copy2(os.path.join(root, file), merged_file_path)

    existing_depth_paths = []
    for root, _, files in os.walk(depth_dir):
        for file in files:
            if file.endswith('masked_color.mp4'):
                existing_depth_paths.append(os.path.join(root, file))
    
    print(f"The length of the depth paths {len(existing_depth_paths)}")

    existing_tracking_paths = []
    for root, _, files in os.walk('data/dexycb_fore_tracking'):
        for file in files:
            if file.endswith('depth.mp4'):
                existing_tracking_paths.append(os.path.join(root, file))
    
    print(f"The length of the tracking paths {len(existing_tracking_paths)}")