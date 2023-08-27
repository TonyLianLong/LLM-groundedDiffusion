
version = "gligen"


from PIL import Image
import torch
import models
from models import model_dict
from models import pipelines
from utils import parse
from prompt import DEFAULT_OVERALL_NEGATIVE_PROMPT
from easydict import EasyDict

verbose = True

assert "gligen" in models.sd_key, models.sd_key

vae, tokenizer, text_encoder, unet, scheduler, dtype = (
    model_dict.vae,
    model_dict.tokenizer,
    model_dict.text_encoder,
    model_dict.unet,
    model_dict.scheduler,
    model_dict.dtype,
)


# Hyperparams
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
H, W = height // 8, width // 8  # size of the latent
num_inference_steps = 50  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
batch_size = 1

run_ind = None


# Note: need to keep the supervision, especially the box corrdinates, corresponds to each other in single object and overall.


def run(spec, gligen_scheduled_sampling_beta=0.4, bg_seed=1):
    """
    so_center_box: using centered box in single object generation
    so_horizontal_center_only: move to the center horizontally only

    align_with_overall_bboxes: Align the center of the mask, latents, and cross-attention with the center of the box in overall bboxes
    horizontal_shift_only: only shift horizontally for the alignment of mask, latents, and cross-attention
    """

    (
        so_prompt_phrase_word_box_list,
        prompt,
        overall_phrases_words_bboxes,
    ) = parse.convert_spec(spec, height, width, verbose=True)
    phrases = [item[0] for item in so_prompt_phrase_word_box_list]
    bboxes = [item[-1] for item in so_prompt_phrase_word_box_list]
    prompts = [prompt]

    negative_prompt = DEFAULT_OVERALL_NEGATIVE_PROMPT
    if "extra_neg_prompt" in spec and spec["extra_neg_prompt"]:
        negative_prompt = (
            spec["extra_neg_prompt"] + ", " + negative_prompt
        )

    
    # Note that so and overall use different negative prompts

    input_embeddings = models.encode_prompts(
        prompts=prompts,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        negative_prompt=negative_prompt,
    )

    
    generator = torch.manual_seed(
        bg_seed
    )  # Seed generator to create the inital latent noise
    latents = models.get_unscaled_latents(
        batch_size, unet.config.in_channels, height, width, generator, dtype
    )

    latents = latents * scheduler.init_noise_sigma

    latents, images = pipelines.generate_gligen(
        model_dict,
        latents,
        input_embeddings,
        num_inference_steps,
        bboxes,
        phrases,
        guidance_scale=guidance_scale,
        gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
    )

    return EasyDict(image=images[0])

    
