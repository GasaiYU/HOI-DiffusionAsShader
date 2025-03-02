import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

prompt = "A hand is trying to grasp the box."
image = load_image(image="val_resources/val_image.jpg")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX1.5-5B-I2V",
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=17,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
    height=384,
    width=640,
).frames[0]

export_to_video(video, "output.mp4", fps=8)
