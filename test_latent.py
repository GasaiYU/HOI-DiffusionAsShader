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
    depth_images_latent = torch.load('data/dexycb_latents_partial_fixed/depth_images_latents/caf08d72-151e-4350-9346-72c5cbfd29fd.pt', weights_only=True)
    normal_images_latent = torch.load('data/dexycb_latents_partial_fixed/normal_images_latents/4a56c924-8d30-4b45-98eb-23ee008603f8.pt', weights_only=True)
    hand_keypoints_images_latent = torch.load('data/dexycb_latents_partial_fixed/hand_keypoints_images_latents/4a56c924-8d30-4b45-98eb-23ee008603f8.pt', weights_only=True)
    image_latents = torch.load('data/dexycb_latents_partial_fixed/image_latents/caf08d72-151e-4350-9346-72c5cbfd29fd.pt', weights_only=True)
    seg_mask_images_latent = torch.load('data/dexycb_latents_partial_fixed/seg_mask_images_latents/4a56c924-8d30-4b45-98eb-23ee008603f8.pt', weights_only=True)
    tracking_images_latent = torch.load('data/dexycb_latents_partial_fixed/tracking_images_latents/4a56c924-8d30-4b45-98eb-23ee008603f8.pt', weights_only=True)
    print(depth_images_latent.shape)
    latent_dict = {
        # "depth_images": depth_images_latent,
        # "normal_images": normal_images_latent,
        # "hand_keypoints_images": hand_keypoints_images_latent,
        "image": image_latents,
        # "seg_mask_images": seg_mask_images_latent,
        # "tracking_images": tracking_images_latent
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
        cv2.imwrite(os.path.join(output_dir, f"{name}_1.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def test_video_latent(
    vae: AutoencoderKLCogVideoX,
    output_dir: str,
):
    depth_video_latent =  torch.load('data/dexycb_latents/depth_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    normal_video_latent = torch.load('data/dexycb_latents/normal_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    hand_keypoints_video_latent = torch.load('data/dexycb_latents/hand_keypoints_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    seg_mask_video_latent = torch.load('data/dexycb_latents/seg_mask_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    tracking_video_latent = torch.load('data/dexycb_latents/tracking_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    video = torch.load('data/dexycb_latents/video_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    print(depth_video_latent.shape)
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

def test_video_latent2rgb(
    vae: AutoencoderKLCogVideoX,
    output_dir: str,
):
    # depth_video_latent =  torch.load('data/dexycb_latents/depth_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    # normal_video_latent = torch.load('data/dexycb_latents/normal_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    # hand_keypoints_video_latent = torch.load('data/dexycb_latents/hand_keypoints_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    # seg_mask_video_latent = torch.load('data/dexycb_latents/seg_mask_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    # tracking_video_latent = torch.load('data/dexycb_latents/tracking_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    video = torch.load('data/dexycb_latents/video_latents/1b278c31-5bd5-4383-b1de-0f0f90cfbfc0.pt', weights_only=True)
    # print(depth_video_latent.shape)
    latent_dict = {
        # "depth_video": depth_video_latent[:,0:1,:,:],
        # "normal_video": normal_video_latent[:,0:1,:,:],
        # "hand_keypoints_video": hand_keypoints_video_latent[:,0:1,:,:],
        # "seg_mask_video": seg_mask_video_latent[:,0:1,:,:],
        # "tracking_video": tracking_video_latent[:,0:1,:,:],
        "image": video[:,0:1,:,:]
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
        cv2.imwrite(os.path.join(output_dir, f"{name}_1.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def test_model_input(
    vae: AutoencoderKLCogVideoX,
    output_dir: str,
):
    depth_latents = torch.load('tmp/latents/depth_latents.pt', weights_only=True)
    hand_keypoints_latents = torch.load('tmp/latents/hand_keypoints_latents.pt', weights_only=True)
    image_latents = torch.load('tmp/latents/image_latents.pt', weights_only=True)
    normal_latents = torch.load('tmp/latents/normal_latents.pt', weights_only=True)
    seg_mask_latents = torch.load('tmp/latents/seg_mask_latents.pt', weights_only=True)
    tracking_latents = torch.load('tmp/latents/tracking_latents.pt', weights_only=True)
    video_latents = torch.load('tmp/latents/video_latents.pt', weights_only=True)
    
    latent_dict = {
        "depth": depth_latents,
        "hand_keypoints": hand_keypoints_latents,
        "normal": normal_latents,
        "seg_mask": seg_mask_latents,
        "tracking": tracking_latents,
    }
    
    scaling_factor = vae.config.scaling_factor
    
    for name, latent in latent_dict.items():
        video_latents, image_latents = latent.chunk(2, dim=2)
        
        video_latents = video_latents.squeeze(0)
        video_latents = video_latents.permute(1, 0, 2, 3) / scaling_factor
        decoded_video_latents = vae.decode(video_latents.unsqueeze(0).to('cuda')).sample.float()
        
        image_latents = image_latents.squeeze(0)
        image_latents = image_latents.permute(1, 0, 2, 3) / scaling_factor
        decoded_image_latents = vae.decode(image_latents[:, :1, ...].unsqueeze(0).to('cuda')).sample.float()
        
        decoded_video_latents = decoded_video_latents.cpu().numpy()
        decoded_video_latents = (decoded_video_latents + 1) / 2
        decoded_video_latents = np.clip(decoded_video_latents, 0, 1)
        decoded_video_latents = (decoded_video_latents * 255).astype(np.uint8)[0]
        decoded_video_latents = np.transpose(decoded_video_latents, (1, 2, 3, 0))
        video_writer = cv2.VideoWriter(os.path.join(output_dir, f"{name}_video.mp4"), fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=15, frameSize=(decoded_video_latents.shape[2], decoded_video_latents.shape[1]))
        for frame in decoded_video_latents:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
        
        decoded_image_latents = decoded_image_latents.cpu().numpy()
        decoded_image_latents = (decoded_image_latents + 1) / 2
        decoded_image_latents = np.clip(decoded_image_latents, 0, 1)
        decoded_image_latents = (decoded_image_latents * 255).astype(np.uint8)[0]
        decoded_image_latents = np.transpose(decoded_image_latents, (1, 2, 3, 0))
        cv2.imwrite(os.path.join(output_dir, f"{name}_image.png"), cv2.cvtColor(decoded_image_latents[0], cv2.COLOR_RGB2BGR))
    pass
        
if __name__ == "__main__":
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        subfolder='vae',
        torch_dtype=torch.bfloat16
    ).to('cuda')

    vae.enable_slicing()
    vae.enable_tiling()
    vae.requires_grad_(False)
    
    output_dir = 'tmp1/test_vae_image_2'
    os.makedirs(output_dir, exist_ok=True)
    
    test_image_latent(vae, output_dir)
    # test_video_latent(vae, output_dir)
    # test_video_latent2rgb(vae,output_dir)
    # test_model_input(vae, output_dir)

