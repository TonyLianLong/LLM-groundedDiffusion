# Original Stable Diffusion (1.5)
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch
import models
from prompt import DEFAULT_OVERALL_NEGATIVE_PROMPT
from easydict import EasyDict

print(f"Using SD: {models.sd_key}")
key = models.sd_key

version = "sd"

pipe = StableDiffusionPipeline.from_pretrained(key).to("cuda")

torch.set_grad_enabled(False)

plot_ind = 0
global_repeat_ind = 0

latent_ratio = 8
num_total_steps = 50
generate_guidance_scale = 7.5

# h, w
image_scale = (512, 512)

bg_negative = DEFAULT_OVERALL_NEGATIVE_PROMPT


def run(prompt, seed=100, extra_neg_prompt=""):
    global global_repeat_ind

    print(f"prompt: {prompt}")
    # keep fg fixed in the first (1-r)T steps and everything flexible in the last rT steps
    generator = torch.Generator("cuda").manual_seed(seed)

    if extra_neg_prompt:
        full_bg_negative = extra_neg_prompt + ", " + bg_negative
    else:
        full_bg_negative = bg_negative

    # mask_image is the background
    images = pipe(
        prompt,
        guidance_scale=generate_guidance_scale,
        negative_prompt=full_bg_negative,
        generator=generator,
    ).images

    return EasyDict(image=images[0])
