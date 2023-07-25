version = "v3.0"

import torch
import numpy as np
import models
import utils
from models import pipelines, sam
from utils import parse, latents
from shared import model_dict, sam_model_dict, DEFAULT_SO_NEGATIVE_PROMPT, DEFAULT_OVERALL_NEGATIVE_PROMPT
import gc

verbose = False
# Accelerates per-box generation
use_fast_schedule = True

vae, tokenizer, text_encoder, unet, dtype = model_dict.vae, model_dict.tokenizer, model_dict.text_encoder, model_dict.unet, model_dict.dtype

model_dict.update(sam_model_dict)


# Hyperparams
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
H, W = height // 8, width // 8 # size of the latent
guidance_scale = 7.5  # Scale for classifier-free guidance

# batch size that is not 1 is not supported
overall_batch_size = 1

# discourage masks with confidence below
discourage_mask_below_confidence = 0.85

# discourage masks with iou (with coarse binarized attention mask) below
discourage_mask_below_coarse_iou = 0.25

run_ind = None


def generate_single_object_with_box_batch(prompts, bboxes, phrases, words, input_latents_list, input_embeddings, 
                                    sam_refine_kwargs, num_inference_steps, gligen_scheduled_sampling_beta=0.3, 
                                    verbose=False, scheduler_key=None, visualize=True, batch_size=None, **kwargs):
    # batch_size=None: does not limit the batch size (pass all input together)
    
    # prompts and words are not used since we don't have cross-attention control in this function
    
    input_latents = torch.cat(input_latents_list, dim=0)
    
    # We need to "unsqueeze" to tell that we have only one box and phrase in each batch item
    bboxes, phrases = [[item] for item in bboxes], [[item] for item in phrases]
    
    input_len = len(bboxes)
    assert len(bboxes) == len(phrases), f"{len(bboxes)} != {len(phrases)}"
    
    if batch_size is None:
        batch_size = input_len
    
    run_times = int(np.ceil(input_len / batch_size))
    mask_selected_list, single_object_pil_images_box_ann, latents_all = [], [], []
    for batch_idx in range(run_times):
        input_latents_batch, bboxes_batch, phrases_batch = input_latents[batch_idx * batch_size:(batch_idx + 1) * batch_size], \
            bboxes[batch_idx * batch_size:(batch_idx + 1) * batch_size], phrases[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        input_embeddings_batch = input_embeddings[0], input_embeddings[1][batch_idx * batch_size:(batch_idx + 1) * batch_size]
        
        _, single_object_images_batch, single_object_pil_images_box_ann_batch, latents_all_batch = pipelines.generate_gligen(
            model_dict, input_latents_batch, input_embeddings_batch, num_inference_steps, bboxes_batch, phrases_batch, gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta, 
            guidance_scale=guidance_scale, return_saved_cross_attn=False,
            return_box_vis=True, save_all_latents=True, batched_condition=True, scheduler_key=scheduler_key, **kwargs
        )
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # `sam_refine_boxes` also calls `empty_cache` so we don't need to explicitly empty the cache again.
        mask_selected, _ = sam.sam_refine_boxes(sam_input_images=single_object_images_batch, boxes=bboxes_batch, model_dict=model_dict, verbose=verbose, **sam_refine_kwargs)
        
        mask_selected_list.append(np.array(mask_selected)[:, 0])
        single_object_pil_images_box_ann.append(single_object_pil_images_box_ann_batch)
        latents_all.append(latents_all_batch)
    
    single_object_pil_images_box_ann, latents_all = sum(single_object_pil_images_box_ann, []), torch.cat(latents_all, dim=1)
    
    # mask_selected_list: List(batch)[List(image)[List(box)[Array of shape (64, 64)]]]
    
    mask_selected = np.concatenate(mask_selected_list, axis=0)
    mask_selected = mask_selected.reshape((-1, *mask_selected.shape[-2:]))
    
    assert mask_selected.shape[0] == input_latents.shape[0], f"{mask_selected.shape[0]} != {input_latents.shape[0]}"
    
    print(mask_selected.shape)
    
    mask_selected_tensor = torch.tensor(mask_selected)
    
    latents_all = latents_all.transpose(0,1)[:,:,None,...]
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return latents_all, mask_selected_tensor, single_object_pil_images_box_ann

def get_masked_latents_all_list(so_prompt_phrase_word_box_list, input_latents_list, so_input_embeddings, verbose=False, **kwargs):
    latents_all_list, mask_tensor_list = [], []
   
    if not so_prompt_phrase_word_box_list:
        return latents_all_list, mask_tensor_list
    
    prompts, bboxes, phrases, words = [], [], [], []

    for prompt, phrase, word, box in so_prompt_phrase_word_box_list:
        prompts.append(prompt)
        bboxes.append(box)
        phrases.append(phrase)
        words.append(word)
    
    latents_all_list, mask_tensor_list, so_img_list = generate_single_object_with_box_batch(prompts, bboxes, phrases, words, input_latents_list, input_embeddings=so_input_embeddings, verbose=verbose, **kwargs)

    return latents_all_list, mask_tensor_list, so_img_list


# Note: need to keep the supervision, especially the box corrdinates, corresponds to each other in single object and overall.

def run(
    spec, bg_seed = 1, overall_prompt_override="", fg_seed_start = 20, frozen_step_ratio=0.4, gligen_scheduled_sampling_beta = 0.3, num_inference_steps = 20,
    so_center_box = False, fg_blending_ratio = 0.1, scheduler_key='dpm_scheduler', so_negative_prompt = DEFAULT_SO_NEGATIVE_PROMPT, overall_negative_prompt = DEFAULT_OVERALL_NEGATIVE_PROMPT, so_horizontal_center_only = True, 
    align_with_overall_bboxes = False, horizontal_shift_only = True, use_autocast = False, so_batch_size = None
):
    """    
    so_center_box: using centered box in single object generation
    so_horizontal_center_only: move to the center horizontally only
    
    align_with_overall_bboxes: Align the center of the mask, latents, and cross-attention with the center of the box in overall bboxes
    horizontal_shift_only: only shift horizontally for the alignment of mask, latents, and cross-attention
    """
    
    print("generation:", spec, bg_seed, fg_seed_start, frozen_step_ratio, gligen_scheduled_sampling_beta)
    
    frozen_step_ratio = min(max(frozen_step_ratio, 0.), 1.)
    frozen_steps = int(num_inference_steps * frozen_step_ratio)

    if True:
        so_prompt_phrase_word_box_list, overall_prompt, overall_phrases_words_bboxes = parse.convert_spec(spec, height, width, verbose=verbose)

    if overall_prompt_override and overall_prompt_override.strip():
        overall_prompt = overall_prompt_override.strip()

    overall_phrases, overall_words, overall_bboxes = [item[0] for item in overall_phrases_words_bboxes], [item[1] for item in overall_phrases_words_bboxes], [item[2] for item in overall_phrases_words_bboxes]

    # The so box is centered but the overall boxes are not (since we need to place to the right place).
    if so_center_box:
        so_prompt_phrase_word_box_list = [(prompt, phrase, word, utils.get_centered_box(bbox, horizontal_center_only=so_horizontal_center_only)) for prompt, phrase, word, bbox in so_prompt_phrase_word_box_list]
        if verbose:
            print(f"centered so_prompt_phrase_word_box_list: {so_prompt_phrase_word_box_list}")
    so_boxes = [item[-1] for item in so_prompt_phrase_word_box_list]

    sam_refine_kwargs = dict(
        discourage_mask_below_confidence=discourage_mask_below_confidence, discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
        height=height, width=width, H=H, W=W
    )
    
    # Note that so and overall use different negative prompts

    with torch.autocast("cuda", enabled=use_autocast):
        so_prompts = [item[0] for item in so_prompt_phrase_word_box_list]
        if so_prompts:
            so_input_embeddings = models.encode_prompts(prompts=so_prompts, tokenizer=tokenizer, text_encoder=text_encoder, negative_prompt=so_negative_prompt, one_uncond_input_only=True)
        else:
            so_input_embeddings = []

        overall_input_embeddings = models.encode_prompts(prompts=[overall_prompt], tokenizer=tokenizer, negative_prompt=overall_negative_prompt, text_encoder=text_encoder)
        
        input_latents_list, latents_bg = latents.get_input_latents_list(
            model_dict, bg_seed=bg_seed, fg_seed_start=fg_seed_start, 
            so_boxes=so_boxes, fg_blending_ratio=fg_blending_ratio, height=height, width=width, verbose=False
        )
        latents_all_list, mask_tensor_list, so_img_list = get_masked_latents_all_list(
            so_prompt_phrase_word_box_list, input_latents_list, 
            gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
            sam_refine_kwargs=sam_refine_kwargs, so_input_embeddings=so_input_embeddings, num_inference_steps=num_inference_steps, scheduler_key=scheduler_key, verbose=verbose, batch_size=so_batch_size,
            fast_after_steps=frozen_steps if use_fast_schedule else None, fast_rate=2
        )

        composed_latents, foreground_indices, offset_list = latents.compose_latents_with_alignment(
            model_dict, latents_all_list, mask_tensor_list, num_inference_steps, 
            overall_batch_size, height, width, latents_bg=latents_bg, 
            align_with_overall_bboxes=align_with_overall_bboxes, overall_bboxes=overall_bboxes,
            horizontal_shift_only=horizontal_shift_only, use_fast_schedule=use_fast_schedule, fast_after_steps=frozen_steps
        )
        
        overall_bboxes_flattened, overall_phrases_flattened = [], []
        for overall_bboxes_item, overall_phrase in zip(overall_bboxes, overall_phrases):
            for overall_bbox in overall_bboxes_item:
                overall_bboxes_flattened.append(overall_bbox)
                overall_phrases_flattened.append(overall_phrase)

        # Generate with composed latents

        # Foreground should be frozen
        frozen_mask = foreground_indices != 0
        
        regen_latents, images = pipelines.generate_gligen(
            model_dict, composed_latents, overall_input_embeddings, num_inference_steps, 
            overall_bboxes_flattened, overall_phrases_flattened, guidance_scale=guidance_scale,
            gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
            frozen_steps=frozen_steps, frozen_mask=frozen_mask, scheduler_key=scheduler_key
        )

        print(f"Generation with spatial guidance from input latents and first {frozen_steps} steps frozen (directly from the composed latents input)")
        print("Generation from composed latents (with semantic guidance)")

        # display(Image.fromarray(images[0]), "img", run_ind)
        
    gc.collect()
    torch.cuda.empty_cache()
        
    return images[0], so_img_list

