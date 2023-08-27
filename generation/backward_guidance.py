version = "backward_guidance"


import torch
import models
from models import model_dict
from models import pipelines
from utils import parse, guidance, latents

from prompt import DEFAULT_OVERALL_NEGATIVE_PROMPT
from easydict import EasyDict

verbose = True

use_fp16 = False

print(f"Using SD: {models.sd_key}")
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

overall_guidance_attn_keys = pipelines.DEFAULT_GUIDANCE_ATTN_KEYS

run_ind = None


# Note: need to keep the supervision, especially the box corrdinates, corresponds to each other in single object and overall.


def run(
    spec,
    bg_seed=1,
    overall_loss_scale=30,
    overall_loss_threshold=0.2,
    overall_max_iter=5,
    overall_max_index_step=10,
):
    """
    so_center_box: using centered box in single object generation
    so_horizontal_center_only: move to the center horizontally only

    align_with_overall_bboxes: Align the center of the mask, latents, and cross-attention with the center of the box in overall bboxes
    horizontal_shift_only: only shift horizontally for the alignment of mask, latents, and cross-attention
    """

    (
        so_prompt_phrase_word_box_list,
        overall_prompt,
        overall_phrases_words_bboxes,
    ) = parse.convert_spec(spec, height, width, verbose=verbose)

    overall_phrases, overall_words, overall_bboxes = (
        [item[0] for item in overall_phrases_words_bboxes],
        [item[1] for item in overall_phrases_words_bboxes],
        [item[2] for item in overall_phrases_words_bboxes],
    )

    overall_negative_prompt = DEFAULT_OVERALL_NEGATIVE_PROMPT
    if "extra_neg_prompt" in spec and spec["extra_neg_prompt"]:
        overall_negative_prompt = (
            spec["extra_neg_prompt"] + ", " + overall_negative_prompt
        )

    # Note that so and overall use different negative prompts

    overall_input_embeddings = models.encode_prompts(
        prompts=[overall_prompt],
        tokenizer=tokenizer,
        negative_prompt=overall_negative_prompt,
        text_encoder=text_encoder,
    )

    generator_bg = torch.manual_seed(
        bg_seed
    )  # Seed generator to create the inital latent noise
    latents_bg = latents.get_scaled_latents(
        batch_size=1,
        in_channels=unet.config.in_channels,
        height=height,
        width=width,
        generator=generator_bg,
        dtype=dtype,
        scheduler=scheduler,
    )

    overall_object_positions, overall_word_token_indices = guidance.get_phrase_indices(
        tokenizer=tokenizer,
        prompt=overall_prompt,
        phrases=overall_phrases,
        words=overall_words,
        verbose=verbose,
        return_word_token_indices=True,
    )

    # This is currently not-shared with the single object one.
    overall_semantic_guidance_kwargs = dict(
        loss_scale=overall_loss_scale,
        loss_threshold=overall_loss_threshold,
        max_iter=overall_max_iter,
        max_index_step=overall_max_index_step,
        # ref_ca comes from the attention map of the word token of the phrase in single object generation, so we apply it only to the word token of the phrase in overall generation.
        ref_ca_word_token_only=True,
        # If a word is not provided, we use the last token.
        ref_ca_last_token_only=True,
        ref_ca_saved_attns=None,
        word_token_indices=overall_word_token_indices,
        guidance_attn_keys=overall_guidance_attn_keys,
        ref_ca_loss_weight=0.5,
        verbose=True,
    )

    img_latents, images = pipelines.generate_semantic_guidance(
        model_dict,
        latents_bg,
        overall_input_embeddings,
        num_inference_steps,
        bboxes=overall_bboxes,
        phrases=overall_phrases,
        object_positions=overall_object_positions,
        guidance_scale=guidance_scale,
        semantic_guidance_kwargs=overall_semantic_guidance_kwargs,
    )

    return EasyDict(image=images[0])
