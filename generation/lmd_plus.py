import torch
import models
import utils
from models import pipelines, sam, model_dict
from utils import parse, guidance, attn, latents, vis
from prompt import (
    DEFAULT_SO_NEGATIVE_PROMPT,
    DEFAULT_OVERALL_NEGATIVE_PROMPT,
)
from easydict import EasyDict

vae, tokenizer, text_encoder, unet, scheduler, dtype = (
    model_dict.vae,
    model_dict.tokenizer,
    model_dict.text_encoder,
    model_dict.unet,
    model_dict.scheduler,
    model_dict.dtype,
)

version = "lmd_plus"

# Hyperparams
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
H, W = height // 8, width // 8  # size of the latent
guidance_scale = 7.5  # Scale for classifier-free guidance

# batch size: set to 1
overall_batch_size = 1

# attn keys for semantic guidance
guidance_attn_keys = pipelines.DEFAULT_GUIDANCE_ATTN_KEYS

# discourage masks with confidence below
discourage_mask_below_confidence = 0.85

# discourage masks with iou (with coarse binarized attention mask) below
discourage_mask_below_coarse_iou = 0.25

offload_cross_attn_to_cpu = False


def generate_single_object_with_box(
    prompt,
    box,
    phrase,
    word,
    input_latents,
    input_embeddings,
    semantic_guidance_kwargs,
    obj_attn_key,
    saved_cross_attn_keys,
    sam_refine_kwargs,
    num_inference_steps,
    gligen_scheduled_sampling_beta=0.3,
    verbose=False,
    visualize=False,
    **kwargs,
):
    bboxes, phrases, words = [box], [phrase], [word]

    if verbose:
        print(f"Getting token map (prompt: {prompt})")

    object_positions, word_token_indices = guidance.get_phrase_indices(
        tokenizer=tokenizer,
        prompt=prompt,
        phrases=phrases,
        words=words,
        return_word_token_indices=True,
        # Since the prompt for single object is from background prompt + object name, we will not have the case of not found
        add_suffix_if_not_found=False,
        verbose=verbose,
    )
    # phrases only has one item, so we select the first item in word_token_indices
    word_token_index = word_token_indices[0]

    if verbose:
        print("word_token_index:", word_token_index)

    # `offload_guidance_cross_attn_to_cpu` will greatly slow down generation
    (
        latents,
        single_object_images,
        saved_attns,
        single_object_pil_images_box_ann,
        latents_all,
    ) = pipelines.generate_gligen(
        model_dict,
        input_latents,
        input_embeddings,
        num_inference_steps,
        bboxes,
        phrases,
        gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
        guidance_scale=guidance_scale,
        return_saved_cross_attn=True,
        semantic_guidance=True,
        semantic_guidance_bboxes=bboxes,
        semantic_guidance_object_positions=object_positions,
        semantic_guidance_kwargs=semantic_guidance_kwargs,
        saved_cross_attn_keys=[obj_attn_key, *saved_cross_attn_keys],
        return_cond_ca_only=True,
        return_token_ca_only=word_token_index,
        offload_cross_attn_to_cpu=offload_cross_attn_to_cpu,
        return_box_vis=True,
        save_all_latents=True,
        dynamic_num_inference_steps=True,
        **kwargs,
    )
    # `saved_cross_attn_keys` kwargs may have duplicates

    utils.free_memory()

    single_object_pil_image_box_ann = single_object_pil_images_box_ann[0]

    if visualize:
        print("Single object image")
        vis.display(single_object_pil_image_box_ann)

    mask_selected, conf_score_selected = sam.sam_refine_box(
        sam_input_image=single_object_images[0],
        box=box,
        model_dict=model_dict,
        verbose=verbose,
        **sam_refine_kwargs,
    )

    mask_selected_tensor = torch.tensor(mask_selected)

    # if visualize:
    #     vis.visualize(mask_selected, "Mask (selected) after resize")
    #     # This is only for visualizations
    #     masked_latents = latents_all * mask_selected_tensor[None, None, None, ...]
    #     vis.visualize_masked_latents(
    #         latents_all, masked_latents, timestep_T=False, timestep_0=True
    #     )

    return (
        latents_all,
        mask_selected_tensor,
        saved_attns,
        single_object_pil_image_box_ann,
    )


def get_masked_latents_all_list(
    so_prompt_phrase_word_box_list,
    input_latents_list,
    so_input_embeddings,
    verbose=False,
    **kwargs,
):
    latents_all_list, mask_tensor_list, saved_attns_list, so_img_list = [], [], [], []

    if not so_prompt_phrase_word_box_list:
        return latents_all_list, mask_tensor_list, saved_attns_list, so_img_list

    so_uncond_embeddings, so_cond_embeddings = so_input_embeddings

    for idx, ((prompt, phrase, word, box), input_latents) in enumerate(
        zip(so_prompt_phrase_word_box_list, input_latents_list)
    ):
        so_current_cond_embeddings = so_cond_embeddings[idx : idx + 1]
        so_current_text_embeddings = torch.cat(
            [so_uncond_embeddings, so_current_cond_embeddings], dim=0
        )
        so_current_input_embeddings = (
            so_current_text_embeddings,
            so_uncond_embeddings,
            so_current_cond_embeddings,
        )

        latents_all, mask_tensor, saved_attns, so_img = generate_single_object_with_box(
            prompt,
            box,
            phrase,
            word,
            input_latents,
            input_embeddings=so_current_input_embeddings,
            verbose=verbose,
            **kwargs,
        )
        latents_all_list.append(latents_all)
        mask_tensor_list.append(mask_tensor)
        saved_attns_list.append(saved_attns)
        so_img_list.append(so_img)

    return latents_all_list, mask_tensor_list, saved_attns_list, so_img_list


def run(
    spec,
    bg_seed=1,
    overall_prompt_override="",
    fg_seed_start=20,
    frozen_step_ratio=0.5,
    num_inference_steps=50,
    loss_scale=5,
    loss_threshold=5.0,
    max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    max_index_step=0,
    overall_loss_scale=5,
    overall_loss_threshold=5.0,
    overall_max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    overall_max_index_step=30,
    so_gligen_scheduled_sampling_beta=0.4,
    overall_gligen_scheduled_sampling_beta=0.4,
    overall_fg_top_p=0.2,
    overall_bg_top_p=0.2,
    overall_fg_weight=1.0,
    overall_bg_weight=4.0,
    ref_ca_loss_weight=2.0,
    so_center_box=False,
    fg_blending_ratio=0.1,
    so_negative_prompt=DEFAULT_SO_NEGATIVE_PROMPT,
    overall_negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT,
    so_horizontal_center_only=True,
    align_with_overall_bboxes=False,
    horizontal_shift_only=True,
    use_fast_schedule=False,
    # Transfer the cross-attention from single object generation (with ref_ca_saved_attns)
    # Use reference cross attention to guide the cross attention in the overall generation
    use_ref_ca=True,
    use_autocast=True,
    verbose=False,
):
    """
    spec: the spec for generation (see generate.py for how to construct a spec)
    bg_seed: background seed
    overall_prompt_override: use custom overall prompt (rather than the object prompt)
    fg_seed_start: each foreground has a seed (fg_seed_start + i), where i is the index of the foreground
    frozen_step_ratio: how many steps should be frozen (as a ratio to inference steps)
    num_inference_steps: number of inference steps
    (overall_)loss_scale: loss scale for per box or overall generation
    (overall_)loss_threshold: loss threshold for per box or overall generation, below which the loss will not be optimized to prevent artifacts
    (overall_)max_iter: max iterations of loss optimization for each step. If scaler, this is applied to all steps.
    (overall_)max_index_step: max index to apply loss optimization to.
    so_gligen_scheduled_sampling_beta and overall_gligen_scheduled_sampling_beta: the guidance steps with GLIGEN
    overall_fg_top_p and overall_bg_top_p: the top P fraction to optimize
    overall_fg_weight and overall_bg_weight: the weight for foreground and background optimization.
    ref_ca_loss_weight: weight for attention transfer (i.e., attention reference loss) to ensure the per-box generation is similar to overall generation in the masked region
    so_center_box: using centered box in single object generation to ensure better spatial control in the generation
    fg_blending_ratio: how much should each foreground initial noise deviate from the background initial noise (and each other)
    so_negative_prompt and overall_negative_prompt: negative prompt for single object (per-box) or overall generation
    so_horizontal_center_only: move to the center horizontally only
    align_with_overall_bboxes: Align the center of the mask, latents, and cross-attention with the center of the box in overall bboxes
    horizontal_shift_only: only shift horizontally for the alignment of mask, latents, and cross-attention
    use_fast_schedule: since the per-box generation, after the steps for latent and attention transfer, is only used by SAM (which does not need to be precise), we skip steps after the steps needed for transfer with a fast schedule.
    use_ref_ca: Use reference cross attention to guide the cross attention in the overall generation
    use_autocast: enable automatic mixed precision (saves memory and makes generation faster)
    Note: attention guidance is disabled for per-box generation by default (`max_index_step` set to 0) because we did not find it improving the results. Attention guidance and reference attention are still enabled for final guidance (overall generation). They greatly improve attribute binding compared to GLIGEN.
    """

    frozen_step_ratio = min(max(frozen_step_ratio, 0.0), 1.0)
    frozen_steps = int(num_inference_steps * frozen_step_ratio)

    print(
        "Key generation settings:",
        spec,
        bg_seed,
        fg_seed_start,
        frozen_step_ratio,
        so_gligen_scheduled_sampling_beta,
        overall_gligen_scheduled_sampling_beta,
        overall_max_index_step,
    )

    (
        so_prompt_phrase_word_box_list,
        overall_prompt,
        overall_phrases_words_bboxes,
    ) = parse.convert_spec(spec, height, width, verbose=verbose)

    if overall_prompt_override and overall_prompt_override.strip():
        overall_prompt = overall_prompt_override.strip()

    overall_phrases, overall_words, overall_bboxes = (
        [item[0] for item in overall_phrases_words_bboxes],
        [item[1] for item in overall_phrases_words_bboxes],
        [item[2] for item in overall_phrases_words_bboxes],
    )

    # The so box is centered but the overall boxes are not (since we need to place to the right place).
    if so_center_box:
        so_prompt_phrase_word_box_list = [
            (
                prompt,
                phrase,
                word,
                utils.get_centered_box(
                    bbox, horizontal_center_only=so_horizontal_center_only
                ),
            )
            for prompt, phrase, word, bbox in so_prompt_phrase_word_box_list
        ]
        if verbose:
            print(
                f"centered so_prompt_phrase_word_box_list: {so_prompt_phrase_word_box_list}"
            )
    so_boxes = [item[-1] for item in so_prompt_phrase_word_box_list]

    if "extra_neg_prompt" in spec and spec["extra_neg_prompt"]:
        so_negative_prompt = spec["extra_neg_prompt"] + ", " + so_negative_prompt
        overall_negative_prompt = (
            spec["extra_neg_prompt"] + ", " + overall_negative_prompt
        )

    semantic_guidance_kwargs = dict(
        loss_scale=loss_scale,
        loss_threshold=loss_threshold,
        max_iter=max_iter,
        max_index_step=max_index_step,
        use_ratio_based_loss=False,
        guidance_attn_keys=guidance_attn_keys,
        verbose=True,
    )

    sam_refine_kwargs = dict(
        discourage_mask_below_confidence=discourage_mask_below_confidence,
        discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
        height=height,
        width=width,
        H=H,
        W=W,
    )

    # if verbose:
    #     vis.visualize_bboxes(
    #         bboxes=[item[-1] for item in so_prompt_phrase_word_box_list], H=H, W=W
    #     )

    # Note that so and overall use different negative prompts

    with torch.autocast("cuda", enabled=use_autocast):
        so_prompts = [item[0] for item in so_prompt_phrase_word_box_list]
        if so_prompts:
            so_input_embeddings = models.encode_prompts(
                prompts=so_prompts,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                negative_prompt=so_negative_prompt,
                one_uncond_input_only=True,
            )
        else:
            so_input_embeddings = []

        input_latents_list, latents_bg = latents.get_input_latents_list(
            model_dict,
            bg_seed=bg_seed,
            fg_seed_start=fg_seed_start,
            so_boxes=so_boxes,
            fg_blending_ratio=fg_blending_ratio,
            height=height,
            width=width,
            verbose=False,
        )

        if use_fast_schedule:
            fast_after_steps = (
                max(frozen_steps, overall_max_index_step)
                if use_ref_ca
                else frozen_steps
            )
        else:
            fast_after_steps = None

        if use_ref_ca or frozen_steps > 0:
            (
                latents_all_list,
                mask_tensor_list,
                saved_attns_list,
                so_img_list,
            ) = get_masked_latents_all_list(
                so_prompt_phrase_word_box_list,
                input_latents_list,
                gligen_scheduled_sampling_beta=so_gligen_scheduled_sampling_beta,
                semantic_guidance_kwargs=semantic_guidance_kwargs,
                obj_attn_key=("down", 2, 1, 0),
                saved_cross_attn_keys=guidance_attn_keys if use_ref_ca else [],
                sam_refine_kwargs=sam_refine_kwargs,
                so_input_embeddings=so_input_embeddings,
                num_inference_steps=num_inference_steps,
                fast_after_steps=fast_after_steps,
                fast_rate=2,
                verbose=verbose,
            )
        else:
            # No per-box guidance
            (latents_all_list, mask_tensor_list, saved_attns_list, so_img_list) = (
                [],
                [],
                [],
                [],
            )

        (
            composed_latents,
            foreground_indices,
            offset_list,
        ) = latents.compose_latents_with_alignment(
            model_dict,
            latents_all_list,
            mask_tensor_list,
            num_inference_steps,
            overall_batch_size,
            height,
            width,
            latents_bg=latents_bg,
            align_with_overall_bboxes=align_with_overall_bboxes,
            overall_bboxes=overall_bboxes,
            horizontal_shift_only=horizontal_shift_only,
            use_fast_schedule=use_fast_schedule,
            fast_after_steps=fast_after_steps,
        )

        # NOTE: need to ensure overall embeddings are generated after the update of overall prompt
        (
            overall_object_positions,
            overall_word_token_indices,
            overall_prompt,
        ) = guidance.get_phrase_indices(
            tokenizer=tokenizer,
            prompt=overall_prompt,
            phrases=overall_phrases,
            words=overall_words,
            verbose=verbose,
            return_word_token_indices=True,
            add_suffix_if_not_found=True,
        )

        overall_input_embeddings = models.encode_prompts(
            prompts=[overall_prompt],
            tokenizer=tokenizer,
            negative_prompt=overall_negative_prompt,
            text_encoder=text_encoder,
        )

        if use_ref_ca:
            # ref_ca_saved_attns has the same hierarchy as bboxes
            ref_ca_saved_attns = []

            flattened_box_idx = 0
            for bboxes in overall_bboxes:
                # bboxes: correspond to a phrase
                ref_ca_current_phrase_saved_attns = []
                for bbox in bboxes:
                    # each individual bbox
                    saved_attns = saved_attns_list[flattened_box_idx]
                    if align_with_overall_bboxes:
                        offset = offset_list[flattened_box_idx]
                        saved_attns = attn.shift_saved_attns(
                            saved_attns,
                            offset,
                            guidance_attn_keys=guidance_attn_keys,
                            horizontal_shift_only=horizontal_shift_only,
                        )
                    ref_ca_current_phrase_saved_attns.append(saved_attns)
                    flattened_box_idx += 1
                ref_ca_saved_attns.append(ref_ca_current_phrase_saved_attns)

        overall_bboxes_flattened, overall_phrases_flattened = [], []
        for overall_bboxes_item, overall_phrase in zip(overall_bboxes, overall_phrases):
            for overall_bbox in overall_bboxes_item:
                overall_bboxes_flattened.append(overall_bbox)
                overall_phrases_flattened.append(overall_phrase)

        # This is currently not-shared with the single object one.
        overall_semantic_guidance_kwargs = dict(
            loss_scale=overall_loss_scale,
            loss_threshold=overall_loss_threshold,
            max_iter=overall_max_iter,
            max_index_step=overall_max_index_step,
            fg_top_p=overall_fg_top_p,
            bg_top_p=overall_bg_top_p,
            fg_weight=overall_fg_weight,
            bg_weight=overall_bg_weight,
            # ref_ca comes from the attention map of the word token of the phrase in single object generation, so we apply it only to the word token of the phrase in overall generation.
            ref_ca_word_token_only=True,
            # If a word is not provided, we use the last token.
            ref_ca_last_token_only=True,
            ref_ca_saved_attns=ref_ca_saved_attns if use_ref_ca else None,
            word_token_indices=overall_word_token_indices,
            guidance_attn_keys=guidance_attn_keys,
            ref_ca_loss_weight=ref_ca_loss_weight,
            use_ratio_based_loss=False,
            verbose=True,
        )

        # Generate with composed latents

        # Foreground should be frozen
        frozen_mask = foreground_indices != 0

        _, images = pipelines.generate_gligen(
            model_dict,
            composed_latents,
            overall_input_embeddings,
            num_inference_steps,
            overall_bboxes_flattened,
            overall_phrases_flattened,
            guidance_scale=guidance_scale,
            gligen_scheduled_sampling_beta=overall_gligen_scheduled_sampling_beta,
            semantic_guidance=True,
            semantic_guidance_bboxes=overall_bboxes,
            semantic_guidance_object_positions=overall_object_positions,
            semantic_guidance_kwargs=overall_semantic_guidance_kwargs,
            frozen_steps=frozen_steps,
            frozen_mask=frozen_mask,
        )

        print(
            f"Generation with spatial guidance from input latents and first {frozen_steps} steps frozen (directly from the composed latents input)"
        )
        print("Generation from composed latents (with semantic guidance)")

    utils.free_memory()

    return EasyDict(image=images[0], so_img_list=so_img_list)
