from moviepy.editor import VideoFileClip, CompositeVideoClip

# 读取两个视频文件
video1 = VideoFileClip("../val_resources/comb_val/val_video.mp4")
video2 = VideoFileClip("../val_resources/comb_val/val_tracking.mp4")

# 确保两个视频的尺寸一致，如果需要，可以缩放其中一个视频
video2 = video2.resize(video1.size)

# 设置第二个视频的透明度（0.0 是完全透明，1.0 是完全不透明）
video2 = video2.set_opacity(0.5)  # 设置不透明度为50%

# 将两个视频叠加在一起
final_video = CompositeVideoClip([video1, video2])

# 输出合成视频
final_video.write_videofile("output_video.mp4", codec="libx264")