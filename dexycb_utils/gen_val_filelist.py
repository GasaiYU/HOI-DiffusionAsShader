import os
import json

def test():
    tracking_training_filelist = 'data/dexycb_filelist/archive/baseline_train_videos.txt'
    depth_training_filelist = 'data/dexycb_filelist/archive/dexycb_depth_training_videos.txt'
    all_videos_filelist = 'data/dexycb_filelist/videos.txt'
    val_videos_filelist = 'data/dexycb_filelist/val_videos.txt'
    sample_num = 200

    os.chdir('..')

    with open(val_videos_filelist, 'r') as f:
        val_videos = f.read().splitlines()
    
    with open(tracking_training_filelist, 'r') as f:
        training_tracking_videos = f.read().splitlines()
    
    with open(depth_training_filelist, 'r') as f:
        training_depth_videos = f.read().splitlines()

    all_trackings_videos = []
    for root, _, files in os.walk('data/dexycb_fore_tracking'):
        for file in files:
            if file.endswith('video_tracking.mp4'):
                all_trackings_videos.append(os.path.join(root, file))

    all_depth_videos = []
    for root, _, files in os.walk('data/dexycb_depth'):
        for file in files:
            if file.endswith('masked_color.mp4'):
                all_depth_videos.append(os.path.join(root, file))
    
    prompts_dict = {}
    with open('data/dexycb_prompt.jsonl', 'r') as f:
        for line in f:
            line_dict = json.loads(line)
            prompts_dict[line_dict['video_path']] = line_dict['response']
    

    val_tracking_paths = []
    val_depth_paths = []
    val_prompts = []
    for val_video_path in val_videos:
        val_tracking_path = os.path.dirname(val_video_path).replace('dexycb_videos', 'dexycb_fore_tracking')
        val_tracking_path = os.path.join(val_tracking_path, 'tracking', 'video_tracking.mp4')
        if val_tracking_path not in training_tracking_videos and val_tracking_path in all_trackings_videos:
            val_tracking_paths.append(val_tracking_path)
            val_prompts.append(prompts_dict[val_video_path])
        
        val_depth_path = os.path.dirname(val_video_path).replace('dexycb_videos', 'dexycb_depth')
        val_depth_path = os.path.join(val_depth_path, 'masked_color.mp4')
        if val_depth_path not in training_depth_videos and val_depth_path in all_depth_videos:
            val_depth_paths.append(val_depth_path)
    
    with open('data/dexycb_filelist/val_trackings.txt', 'w') as f:
        for path in val_tracking_paths:
            f.write(path + '\n')
    
    with open('data/dexycb_filelist/val_depths.txt', 'w') as f:
        for path in val_depth_paths:
            f.write(path + '\n')
        
    with open('data/dexycb_filelist/val_prompts.txt', 'w') as f:
        for prompt in val_prompts:
            f.write(prompt + '\n')
        
    print(f"The length of the tracking paths {len(val_tracking_paths)}")
    print(f"The length of the depth paths {len(val_depth_paths)}")
    # tracking_videos = []
    # with open(tracking_training_filelist, 'r') as f:
    #     for line in f:
    #         tracking_videos.append(line.strip())

    # depth_videos = []
    # with open(depth_training_filelist, 'r') as f:
    #     for line in f:
    #         depth_videos.append(line.strip())
    
    # training_videos = set(tracking_videos) | set(depth_videos)

    # val_videos = []
    # with open(all_videos_filelist, 'r') as f:
    #     for line in f:
    #         video = line.strip()
    #         if video not in training_videos:
    #             val_videos.append(video)
    
    # val_videos = random.sample(val_videos, 200)

    # with open('data/dexycb_filelist/val_videos.txt', 'w') as f:
    #     for video in val_videos:
    #         f.write(video + '\n')

    pass

if __name__ == "__main__":
    val_video_filelist = '../data/dexycb_filelist/val_videos.txt'
    with open(val_video_filelist, 'r') as f:
        val_videos = f.read().splitlines()
    
    normal_videos = []
    for video in val_videos:
        normal_video = video.replace('dexycb_videos', 'dexycb_normal/dexycb_videos')
        if os.path.exists(os.path.join('..',normal_video)):
            normal_videos.append(normal_video)
        else:
            print(f"Normal video {normal_video} does not exist")
    
    with open('../data/dexycb_filelist/val_normals.txt', 'w') as f:
        for video in normal_videos:
            f.write(video + '\n')