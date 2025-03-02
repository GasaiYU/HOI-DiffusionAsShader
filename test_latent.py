from diffusers import AutoencoderKLCogVideoX
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import torch
import numpy as np

import cv2
import os

def test_image_latent(
    vae: AutoencoderKLCogVideoX,
    output_dir: str,
):
    depth_images_latent = torch.load('data/dexycb_latents/tmp/depth_images_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    normal_images_latent = torch.load('data/dexycb_latents/tmp/normal_images_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    hand_keypoints_images_latent = torch.load('data/dexycb_latents/tmp/hand_keypoints_images_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    image_latents = torch.load('data/dexycb_latents/tmp/image_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    seg_mask_images_latent = torch.load('data/dexycb_latents/tmp/seg_mask_images_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    tracking_images_latent = torch.load('data/dexycb_latents/tmp/tracking_images_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    
    latent_dict = {
        "depth_images": depth_images_latent,
        "normal_images": normal_images_latent,
        "hand_keypoints_images": hand_keypoints_images_latent,
        "image": image_latents,
        "seg_mask_images": seg_mask_images_latent,
        "tracking_images": tracking_images_latent
    }
        
    for name, latent in latent_dict.items():    
        post_latent = DiagonalGaussianDistribution(latent.unsqueeze(0).to('cuda').to(torch.bfloat16))
        decoded_latent = vae.decode(post_latent.sample()).sample.float()
        
        decoded_latent = decoded_latent.cpu().numpy()
        decoded_latent = (decoded_latent + 1) / 2
        decoded_latent = np.clip(decoded_latent, 0, 1)
        decoded_latent = (decoded_latent * 255).astype(np.uint8)[0]
        decoded_latent = np.transpose(decoded_latent, (1, 2, 3, 0))

        image = decoded_latent[0]
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def test_video_latent(
    vae: AutoencoderKLCogVideoX,
    output_dir: str,
):
    depth_video_latent =  torch.load('data/dexycb_latents/tmp/depth_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    normal_video_latent = torch.load('data/dexycb_latents/tmp/normal_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    hand_keypoints_video_latent = torch.load('data/dexycb_latents/tmp/hand_keypoints_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    seg_mask_video_latent = torch.load('data/dexycb_latents/tmp/seg_mask_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    tracking_video_latent = torch.load('data/dexycb_latents/tmp/tracking_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    video = torch.load('data/dexycb_latents/tmp/video_latents/0/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    
    latent_dict = {
        "depth_video": depth_video_latent,
        "normal_video": normal_video_latent,
        "hand_keypoints_video": hand_keypoints_video_latent,
        "seg_mask_video": seg_mask_video_latent,
        "tracking_video": tracking_video_latent,
        "video": video
    }
    
    for name, latent in latent_dict.items():
        post_latent = DiagonalGaussianDistribution(latent.unsqueeze(0).to('cuda').to(torch.bfloat16))
        decoded_latent = vae.decode(post_latent.sample()).sample.float()
        
        decoded_latent = decoded_latent.cpu().numpy()
        decoded_latent = (decoded_latent + 1) / 2
        decoded_latent = np.clip(decoded_latent, 0, 1)
        decoded_latent = (decoded_latent * 255).astype(np.uint8)[0]
        decoded_latent = np.transpose(decoded_latent, (1, 2, 3, 0))

        video_writer = cv2.VideoWriter(os.path.join(output_dir, f"{name}.mp4"), fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=15, frameSize=(decoded_latent.shape[2], decoded_latent.shape[1]))
        for frame in decoded_latent:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
        
        
if __name__ == "__main__":
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        subfolder='vae',
        torch_dtype=torch.bfloat16
    ).to('cuda')

    vae.enable_slicing()
    vae.enable_tiling()
    vae.requires_grad_(False)
    
    output_dir = 'tmp'
    os.makedirs(output_dir, exist_ok=True)
    
    test_image_latent(vae, output_dir)
    test_video_latent(vae, output_dir)

