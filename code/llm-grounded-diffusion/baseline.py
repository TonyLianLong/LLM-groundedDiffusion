# Original Stable Diffusion (1.4)

import torch
import models
from models import pipelines
from shared import model_dict, DEFAULT_OVERALL_NEGATIVE_PROMPT
import gc

vae, tokenizer, text_encoder, unet, scheduler, dtype = model_dict.vae, model_dict.tokenizer, model_dict.text_encoder, model_dict.unet, model_dict.scheduler, model_dict.dtype

torch.set_grad_enabled(False)

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
guidance_scale = 7.5  # Scale for classifier-free guidance
batch_size = 1

# h, w
image_scale = (512, 512)

bg_negative = DEFAULT_OVERALL_NEGATIVE_PROMPT

# Using dpm scheduler by default
def run(prompt, scheduler_key='dpm_scheduler', bg_seed=1, num_inference_steps=20):
    print(f"prompt: {prompt}")
    generator = torch.manual_seed(bg_seed)
    
    prompts = [prompt]
    input_embeddings = models.encode_prompts(prompts=prompts, tokenizer=tokenizer, text_encoder=text_encoder, negative_prompt=bg_negative)

    latents = models.get_unscaled_latents(batch_size, unet.config.in_channels, height, width, generator, dtype)

    latents = latents * scheduler.init_noise_sigma

    pipelines.gligen_enable_fuser(model_dict['unet'], enabled=False)
    _, images = pipelines.generate(
        model_dict, latents, input_embeddings, num_inference_steps,  
        guidance_scale=guidance_scale, scheduler_key=scheduler_key
    )
    
    gc.collect()
    torch.cuda.empty_cache()

    return images[0]