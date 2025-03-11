import cv2
import numpy as np

import os

from tqdm import tqdm


def alpha_blend(video_file_1, video_file_2):
    # Open video files
    cap1 = cv2.VideoCapture(video_file_1)
    cap2 = cv2.VideoCapture(video_file_2)

    # Get video properties
    fps = cap1.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # # Initialize video writer (output)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec
    # out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Define alpha value (transparency) for blending
    alpha = 0.5  # 0.0 (only video1) to 1.0 (only video2)
    out_frames = []

    while True:
        # Read frames from both videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Check if we have reached the end of either video
        if not ret1 or not ret2:
            break

        # Resize frame2 to match the size of frame1
        frame2 = cv2.resize(frame2, (frame_width, frame_height))

        # Alpha blend the two frames
        blended_frame = cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)

        # Write the blended frame to the output video
        out_frames.append(blended_frame)

    # Release resources
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    return np.stack(out_frames, axis=0)

def main(data_root='../data/dexycb_latents', out_dir='../data/preprocess/blended_data'):
    os.makedirs(out_dir, exist_ok=True)
    condition_keys = ['depth', 'hand_keypoints', 'seg_mask', 'tracking']
    for path in tqdm(os.listdir(os.path.join(data_root, 'videos'))):
        merged_video = []     
        video_path = os.path.join(data_root, 'videos', path)
        for condition_key in condition_keys:
            condition_path = os.path.join(data_root, condition_key, path)
            alpha_blend_video = alpha_blend(video_path, condition_path)
            merged_video.append(alpha_blend_video)
        
        merged_video = np.concatenate(merged_video, axis=2)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(os.path.join(out_dir, path), fourcc, 7, (merged_video.shape[2], merged_video.shape[1]))

        for merged_frame in merged_video:
            output.write(merged_frame)
        output.release()

    pass


if __name__ == "__main__":
    main()