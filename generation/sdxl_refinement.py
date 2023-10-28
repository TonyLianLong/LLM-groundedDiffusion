from diffusers import StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image

# This is adapted to SDXL since it often generates styles that we don't want. If you want to generate these styles, please change the negative prompt.
sdxl_negative_prompt = "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"

pipe = None

def init(offload_model=True):
    global pipe
    
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )

    if offload_model:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

def refine(image, spec, refine_seed, refinement_step_ratio=0.5):
    overall_prompt = spec["prompt"]
    extra_neg_prompt = spec["extra_neg_prompt"]
    g = torch.manual_seed(refine_seed)
    image = Image.fromarray(image).resize((1024, 1024), Image.LANCZOS)
    negative_prompt = extra_neg_prompt + ", " + sdxl_negative_prompt
    output = pipe(overall_prompt, image=image, negative_prompt=negative_prompt, strength=refinement_step_ratio, generator=g).images[0]
    
    return output
