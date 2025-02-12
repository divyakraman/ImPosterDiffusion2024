import requests
from PIL import Image
from io import BytesIO
import torch
import os
from diffusers import DiffusionPipeline, DDIMScheduler
import PIL
import cv2
import numpy as np 
#from scipy import ndimage #rotation angle in degree
#import matplotlib.pyplot as plt 

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
torch.hub.set_dir('')

height = 512
width = 512

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
        safety_checker=None,
    use_auth_token=False,
    custom_pipeline='./models/imposter_lora', cache_dir = 'dir_name',
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
).to(device)


generator = torch.Generator("cuda").manual_seed(0)
seed = 0
num_inference_steps = 50 #50

prompt1 = "A teddy bear" 
prompt2 = "A man playing guitar"  
prompt3 = "A teddy bear playing guitar"

source_image = PIL.Image.open('').convert("RGB")
source_image = expand2square(source_image, (0,0,0))
#source_image = source_image.resize((256, 256))
source_image = source_image.resize((512, 512)) 

driving_recon_image = PIL.Image.open('').convert("RGB")
driving_recon_image = expand2square(driving_recon_image, (0,0,0))
#driving_recon_image = driving_recon_image.resize((256, 256))
driving_recon_image = driving_recon_image.resize((512, 512)) 

res = pipe.train(
    prompt1 = prompt1, prompt2 = prompt2, prompt3 = prompt3,
    source_image=source_image,
    driving_recon_image = driving_recon_image,
    generator=generator, text_embedding_optimization_steps = 500,
        model_fine_tuning_optimization_steps = 500)

K = 6
amp_lr = 1e-2 * 1e-4 
phase_lr = 1e-2 * 1e-1 

res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=num_inference_steps, height=height, width=width, K=K, amp_lr=amp_lr, phase_lr=phase_lr)
image = res.images[0]
image.save('./result1.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=num_inference_steps, height=height, width=width, K=K, amp_lr=amp_lr, phase_lr=phase_lr)
image = res.images[0]
image.save('./result2.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=num_inference_steps, height=height, width=width, K=K, amp_lr=amp_lr, phase_lr=phase_lr)
image = res.images[0]
image.save('./result3.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=num_inference_steps, height=height, width=width, K=K, amp_lr=amp_lr, phase_lr=phase_lr)
image = res.images[0]
image.save('./result4.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=num_inference_steps, height=height, width=width, K=K, amp_lr=amp_lr, phase_lr=phase_lr)
image = res.images[0]
image.save('./result5.png')

