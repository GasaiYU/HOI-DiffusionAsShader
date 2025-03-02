import os

import json

if __name__ == "__main__":
    os.chdir('..')

    depth_paths = []
    for root, dirs, files in os.walk('data/dexycb_depth'):
        for file in files:
            if file.endswith('.mp4'):
                depth_paths.append(os.path.dirname(os.path.join(root, file)))
    depth_paths = list(set(depth_paths)) # remove duplicates

    normal_paths = []
    for root, dirs, files in os.walk('data/dexycb_normal'):
        for file in files:
            if file.endswith('.mp4'):
                normal_paths.append(os.path.dirname(os.path.join(root, file)))
    normal_paths = list(set(normal_paths)) # remove duplicates

    tracking_map_paths = []
    for root, dirs, files in os.walk('data/dexycb_fore_tracking'):
        for file in files:
            if file.endswith('.mp4'):
                tracking_map_paths.append(os.path.dirname(os.path.join(root, file)))
    tracking_map_paths = list(set(tracking_map_paths)) # remove duplicates

    val_depth_paths = []
    with open('data/dexycb_filelist/val_depths.txt', 'r') as f:
        for line in f.readlines():
            val_depth_paths.append(os.path.dirname(line.strip()))
    val_depth_paths = list(set(val_depth_paths)) # remove duplicates
    
    val_tracking_paths = []
    with open('data/dexycb_filelist/val_trackings.txt', 'r') as f:
        for line in f.readlines():
            val_tracking_paths.append(os.path.dirname(line.strip()))
    val_tracking_paths = list(set(val_tracking_paths)) # remove duplicates

    prompts_list = []
    videos_list = []
    depths_list = []
    normals_list = []
    tracking_maps_list = []
    labels_list = []

    missing_list = []
    with open('data/dexycb_prompt.jsonl') as f:
        for line in f.readlines():
            prompt_line = json.loads(line)
            video_path = prompt_line['video_path']

            depth_path = os.path.dirname(video_path).replace('dexycb_videos', 'dexycb_depth')
            normal_path = os.path.dirname(video_path).replace('dexycb_videos', 'dexycb_normal/dexycb_videos')
            tracking_map_path = os.path.dirname(video_path).replace('dexycb_videos', 'dexycb_fore_tracking')
            label_path = os.path.dirname(video_path).replace('dexycb_videos', 'dexycb')

            if depth_path in depth_paths and depth_path not in val_depth_paths:
                prompts_list.append(prompt_line['response'])    
                videos_list.append(prompt_line['video_path'])
                depths_list.append(os.path.join(depth_path, 'masked_color.mp4'))
                labels_list.append(label_path)
            if normal_path in normal_paths:
                normals_list.append(os.path.join(normal_path, 'video.mp4'))
            if os.path.join(tracking_map_path, 'tracking') in tracking_map_paths and os.path.join(tracking_map_path, 'tracking') not in val_tracking_paths:
                tracking_maps_list.append(os.path.join(tracking_map_path, 'tracking', 'video_tracking.mp4'))
        
    # with open('data/dexycb_filelist/missing_tracking.txt', 'w') as f:
    #     for i in range(len(missing_list)):
    #         f.write(missing_list[i] + '\n')
    
    with open('data/dexycb_filelist/training/training_videos.txt', 'w') as f:
        for i in range(len(videos_list)):
            f.write(videos_list[i] + '\n')  
    
    with open('data/dexycb_filelist/training/training_prompts.txt', 'w') as f:
        for i in range(len(prompts_list)):
            f.write(prompts_list[i] + '\n')
    
    with open('data/dexycb_filelist/training/training_depths.txt', 'w') as f:
        for i in range(len(depths_list)):
            f.write(os.path.join(depths_list[i]) + '\n')
    
    with open('data/dexycb_filelist/training/training_normals.txt', 'w') as f:
        for i in range(len(normals_list)):
            f.write(os.path.join(normals_list[i]) + '\n')
    
    with open('data/dexycb_filelist/training/training_trackings.txt', 'w') as f:
        for i in range(len(tracking_maps_list)):
            f.write(os.path.join(tracking_maps_list[i]) + '\n')
    
    with open('data/dexycb_filelist/training/training_labels.txt', 'w') as f:
        for i in range(len(labels_list)):
            f.write(os.path.join(labels_list[i]) + '\n')
    
    pass