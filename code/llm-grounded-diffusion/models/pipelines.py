import torch
from tqdm import tqdm
import utils
from PIL import Image
import gc
import numpy as np
from .attention import GatedSelfAttentionDense
from .models import process_input_embeddings, torch_device

@torch.no_grad()
def encode(model_dict, image, generator):
    """
    image should be a PIL object or numpy array with range 0 to 255
    """
    
    vae, dtype = model_dict.vae, model_dict.dtype
    
    if isinstance(image, Image.Image):
        w, h = image.size
        assert w % 8 == 0 and h % 8 == 0, f"h ({h}) and w ({w}) should be a multiple of 8"
        # w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        # image = np.array(image.resize((w, h), resample=Image.Resampling.LANCZOS))[None, :]
        image = np.array(image)
    
    if isinstance(image, np.ndarray):
        assert image.dtype == np.uint8, f"Should have dtype uint8 (dtype: {image.dtype})"
        image = image.astype(np.float32) / 255.0
        image = image[None, ...]
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    
    assert isinstance(image, torch.Tensor), f"type of image: {type(image)}"
    
    image = image.to(device=torch_device, dtype=dtype)
    latents = vae.encode(image).latent_dist.sample(generator)
    
    latents = vae.config.scaling_factor * latents

    return latents

@torch.no_grad()
def decode(vae, latents):
    # scale and decode the image latents with vae
    scaled_latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(scaled_latents).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    
    return images

@torch.no_grad()
def generate(model_dict, latents, input_embeddings, num_inference_steps, guidance_scale = 7.5, no_set_timesteps=False, scheduler_key='dpm_scheduler'):
    vae, tokenizer, text_encoder, unet, scheduler, dtype = model_dict.vae, model_dict.tokenizer, model_dict.text_encoder, model_dict.unet, model_dict[scheduler_key], model_dict.dtype
    text_embeddings, uncond_embeddings, cond_embeddings = input_embeddings
    
    if not no_set_timesteps:
        scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    images = decode(vae, latents)
    
    ret = [latents, images]

    return tuple(ret)

def gligen_enable_fuser(unet, enabled=True):
    for module in unet.modules():
        if isinstance(module, GatedSelfAttentionDense):
            module.enabled = enabled

def prepare_gligen_condition(bboxes, phrases, dtype, tokenizer, text_encoder, num_images_per_prompt):
    batch_size = len(bboxes)
    
    assert len(phrases) == len(bboxes)
    max_objs = 30
    
    n_objs = min(max([len(bboxes_item) for bboxes_item in bboxes]), max_objs)
    boxes = torch.zeros((batch_size, max_objs, 4), device=torch_device, dtype=dtype)
    phrase_embeddings = torch.zeros((batch_size, max_objs, 768), device=torch_device, dtype=dtype)
    # masks is a 1D tensor deciding which of the enteries to be enabled
    masks = torch.zeros((batch_size, max_objs), device=torch_device, dtype=dtype)
    
    if n_objs > 0:
        for idx, (bboxes_item, phrases_item) in enumerate(zip(bboxes, phrases)):
            # the length of `bboxes_item` could be smaller than `n_objs` because n_objs takes the max of item length
            bboxes_item = torch.tensor(bboxes_item[:n_objs])
            boxes[idx, :bboxes_item.shape[0]] = bboxes_item

            tokenizer_inputs = tokenizer(phrases_item[:n_objs], padding=True, return_tensors="pt").to(torch_device)
            _phrase_embeddings = text_encoder(**tokenizer_inputs).pooler_output
            phrase_embeddings[idx, :_phrase_embeddings.shape[0]] = _phrase_embeddings
            assert bboxes_item.shape[0] == _phrase_embeddings.shape[0], f"{bboxes_item.shape[0]} != {_phrase_embeddings.shape[0]}"
            
            masks[idx, :bboxes_item.shape[0]] = 1

    # Classifier-free guidance
    repeat_times = num_images_per_prompt * 2
    condition_len = batch_size * repeat_times

    boxes = boxes.repeat(repeat_times, 1, 1)
    phrase_embeddings = phrase_embeddings.repeat(repeat_times, 1, 1)
    masks = masks.repeat(repeat_times, 1)
    masks[:condition_len // 2] = 0
    
    # print("shapes:", boxes.shape, phrase_embeddings.shape, masks.shape)
    
    return boxes, phrase_embeddings, masks, condition_len

@torch.no_grad()
def generate_gligen(model_dict, latents, input_embeddings, num_inference_steps, bboxes, phrases, num_images_per_prompt=1, gligen_scheduled_sampling_beta: float = 0.3, guidance_scale=7.5, 
    frozen_steps=20, frozen_mask=None,
    return_saved_cross_attn=False, saved_cross_attn_keys=None, return_cond_ca_only=False, return_token_ca_only=None, 
    offload_cross_attn_to_cpu=False, offload_latents_to_cpu=True,
    return_box_vis=False, show_progress=True, save_all_latents=False, scheduler_key='dpm_scheduler', batched_condition=False):
    """
    The `bboxes` should be a list, rather than a list of lists (one box per phrase, we can have multiple duplicated phrases).
    """
    vae, tokenizer, text_encoder, unet, scheduler, dtype = model_dict.vae, model_dict.tokenizer, model_dict.text_encoder, model_dict.unet, model_dict[scheduler_key], model_dict.dtype
    
    text_embeddings, _, cond_embeddings = process_input_embeddings(input_embeddings)
    
    if latents.dim() == 5:
        # latents_all from the input side, different from the latents_all to be saved
        latents_all_input = latents
        latents = latents[0]
    else:
        latents_all_input = None
    
    # Just in case that we have in-place ops
    latents = latents.clone()
    
    if save_all_latents:
        # offload to cpu to save space
        if offload_latents_to_cpu:
            latents_all = [latents.cpu()]
        else:
            latents_all = [latents]
    
    scheduler.set_timesteps(num_inference_steps)
    
    if frozen_mask is not None:
        frozen_mask = frozen_mask.to(dtype=dtype).clamp(0., 1.)

    # 5.1 Prepare GLIGEN variables
    if not batched_condition:
        # Add batch dimension to bboxes and phrases
        bboxes, phrases = [bboxes], [phrases]
    
    boxes, phrase_embeddings, masks, condition_len = prepare_gligen_condition(bboxes, phrases, dtype, tokenizer, text_encoder, num_images_per_prompt)
    
    if return_saved_cross_attn:
        saved_attns = []
    
    main_cross_attention_kwargs = {
        'offload_cross_attn_to_cpu': offload_cross_attn_to_cpu,
        'return_cond_ca_only': return_cond_ca_only,
        'return_token_ca_only': return_token_ca_only,
        'save_keys': saved_cross_attn_keys,
        'gligen': {
            'boxes': boxes,
            'positive_embeddings': phrase_embeddings,
            'masks': masks
        }
    }
    
    timesteps = scheduler.timesteps

    num_grounding_steps = int(gligen_scheduled_sampling_beta * len(timesteps))
    gligen_enable_fuser(unet, True)

    for index, t in enumerate(tqdm(timesteps, disable=not show_progress)):
        # Scheduled sampling
        if index == num_grounding_steps:
            gligen_enable_fuser(unet, False)
        
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        main_cross_attention_kwargs['save_attn_to_dict'] = {}

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, 
                            cross_attention_kwargs=main_cross_attention_kwargs).sample
        
        if return_saved_cross_attn:
            saved_attns.append(main_cross_attention_kwargs['save_attn_to_dict'])
            
            del main_cross_attention_kwargs['save_attn_to_dict']

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        if frozen_mask is not None and index < frozen_steps:
            latents = latents_all_input[index+1] * frozen_mask + latents * (1. - frozen_mask)
        
        if save_all_latents:
            if offload_latents_to_cpu:
                latents_all.append(latents.cpu())
            else:
                latents_all.append(latents)

    # Turn off fuser for typical SD
    gligen_enable_fuser(unet, False)
    images = decode(vae, latents)
    
    ret = [latents, images]
    if return_saved_cross_attn:
        ret.append(saved_attns)
    if return_box_vis:
        pil_images = [utils.draw_box(Image.fromarray(image), bboxes_item, phrases_item) for image, bboxes_item, phrases_item in zip(images, bboxes, phrases)]
        ret.append(pil_images)
    if save_all_latents:
        latents_all = torch.stack(latents_all, dim=0)
        ret.append(latents_all)
    
    return tuple(ret)

