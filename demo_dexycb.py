import os
import sys
import argparse
from PIL import Image
project_root = os.path.dirname(os.path.abspath(__file__))
# try:
#     sys.path.append(os.path.join(project_root, "submodules/MoGe"))
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
# except:
#     print("Warning: MoGe not found, motion transfer will not be applied")
    
import torch
import numpy as np
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
from diffusers.utils import load_image, load_video

from models.pipelines import DiffusionAsShaderPipeline

from tqdm import tqdm


def load_media(media_path, max_frames=49, transform=None):
    """Load video or image frames and convert to tensor
    
    Args:
        media_path (str): Path to video or image file
        max_frames (int): Maximum number of frames to load
        transform (callable): Transform to apply to frames
        
    Returns:
        Tuple[torch.Tensor, float]: Video tensor [T,C,H,W] and FPS
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])
    
    # Determine if input is video or image based on extension
    ext = os.path.splitext(media_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov']
    
    if is_video:
        frames = load_video(media_path)
        fps = len(frames) / VideoFileClip(media_path).duration
    else:
        # Handle image as single frame
        image = load_image(media_path)
        frames = [image]
        fps = 8  # Default fps for images
    
    # Ensure we have exactly max_frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        last_frame = frames[-1]
        while len(frames) < max_frames:
            frames.append(last_frame.copy())
            
    # Convert frames to tensor
    video_tensor = torch.stack([transform(frame) for frame in frames])
    
    return video_tensor, fps, is_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filelist', type=str, default=None, help='Path to input video/image')
    parser.add_argument('--prompt_filelist', type=str, required=True, help='Repaint prompt')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--transformer_path', type=str, default=None, help='Path to transformer checkpoint')
    parser.add_argument('--tracking_filelist', type=str, default=None, help='Path to tracking video, if provided, camera motion and object manipulation will not be applied')
    args = parser.parse_args()

    # Initialize pipeline
    das = DiffusionAsShaderPipeline(gpu_id=args.gpu, output_dir=args.output_dir, model_path=args.checkpoint_path, transformer_path=args.transformer_path)

    with open(args.prompt_filelist, 'r') as f:
        prompt_list = f.readlines()
    prompt_list = [prompt.strip() for prompt in prompt_list]

    with open(args.input_filelist, 'r') as f:
        input_list = f.readlines()
    input_list = [input.strip() for input in input_list]

    with open(args.tracking_filelist, 'r') as f:
        tracking_list = f.readlines()
    tracking_list = [tracking.strip() for tracking in tracking_list]

    for i in tqdm(range(len(input_list))):
        input_path = input_list[i]
        prompt = prompt_list[i]
        tracking_path = tracking_list[i]

        # Load input video/image
        video_tensor, fps, is_video = load_media(input_path)

        # Repaint first frame if requested
        repaint_img_tensor = None

        # Generate tracking if not provided
        tracking_tensor, _, _ = load_media(tracking_path)

        das.apply_tracking(
            video_tensor=video_tensor,
            fps=8,
            tracking_tensor=tracking_tensor,
            img_cond_tensor=repaint_img_tensor,
            prompt=prompt,
        )