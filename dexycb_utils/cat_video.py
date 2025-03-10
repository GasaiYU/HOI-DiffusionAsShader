import cv2
import numpy as np


video_paths = [
    '../exps/cogshader_HOI_comb_latents_no_depth_100000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/validation_ep43_0th_Initially,_a_man_in_a_dar.mp4',
    '../val_resources/comb_val/val_video.mp4',
    '../val_resources/comb_val/val_tracking.mp4',
    '../val_resources/comb_val/val_normal.mp4',
    '../val_resources/comb_val/val_seg_mask.mp4',
    '../val_resources/comb_val/val_hand_keypoints.mp4',
]

all_frames = []
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (720, 480)))
    all_frames.append(np.stack(frames[:49], axis=0))
    cap.release()

all_frames_line1 = np.concatenate(all_frames[:3], axis=2)
all_frames_line2 = np.concatenate(all_frames[3:], axis=2)
all_frames = [all_frames_line1, all_frames_line2]
all_frames = np.concatenate(all_frames, axis=1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 15, (all_frames.shape[2], all_frames.shape[1]))

for frame in all_frames:
    out.write(frame)

out.release()