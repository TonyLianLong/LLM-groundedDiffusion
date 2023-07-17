import torch
import numpy as np
from . import utils
from utils import torch_device
import matplotlib.pyplot as plt

def get_unscaled_latents(batch_size, in_channels, height, width, generator, dtype):
    """
    in_channels: often obtained with `unet.config.in_channels`
    """
    # Obtain with torch.float32 and cast to float16 if needed
    # Directly obtaining latents in float16 will lead to different latents
    latents_base = torch.randn(
        (batch_size, in_channels, height // 8, width // 8),
        generator=generator, dtype=dtype
    ).to(torch_device, dtype=dtype)
    
    return latents_base

def get_scaled_latents(batch_size, in_channels, height, width, generator, dtype, scheduler):
    latents_base = get_unscaled_latents(batch_size, in_channels, height, width, generator, dtype)
    latents_base = latents_base * scheduler.init_noise_sigma
    return latents_base 

def blend_latents(latents_bg, latents_fg, fg_mask, fg_blending_ratio=0.01):
    """
    in_channels: often obtained with `unet.config.in_channels`
    """
    assert not torch.allclose(latents_bg, latents_fg), "latents_bg should be independent with latents_fg"
    
    dtype = latents_bg.dtype
    latents = latents_bg * (1. - fg_mask) + (latents_bg * np.sqrt(1. - fg_blending_ratio) + latents_fg * np.sqrt(fg_blending_ratio)) * fg_mask
    latents = latents.to(dtype=dtype)

    return latents

@torch.no_grad()
def compose_latents(model_dict, latents_all_list, mask_tensor_list, num_inference_steps, overall_batch_size, height, width, latents_bg=None, bg_seed=None, compose_box_to_bg=True):
    unet, scheduler, dtype = model_dict.unet, model_dict.scheduler, model_dict.dtype
    
    if latents_bg is None:
        generator = torch.manual_seed(bg_seed)  # Seed generator to create the inital latent noise
        latents_bg = get_scaled_latents(overall_batch_size, unet.config.in_channels, height, width, generator, dtype, scheduler)
    
    # Other than t=T (idx=0), we only have masked latents. This is to prevent accidentally loading from non-masked part. Use same mask as the one used to compose the latents.
    composed_latents = torch.zeros((num_inference_steps + 1, *latents_bg.shape), dtype=dtype)
    composed_latents[0] = latents_bg
    
    foreground_indices = torch.zeros(latents_bg.shape[-2:], dtype=torch.long)
    
    mask_size = np.array([mask_tensor.sum().item() for mask_tensor in mask_tensor_list])
    # Compose the largest mask first
    mask_order = np.argsort(-mask_size)
    
    if compose_box_to_bg:
        # This has two functionalities: 
        # 1. copies the right initial latents from the right place (for centered so generation), 2. copies the right initial latents (since we have foreground blending) for centered/original so generation.
        for mask_idx in mask_order:
            latents_all, mask_tensor = latents_all_list[mask_idx], mask_tensor_list[mask_idx]
            
            # Note: need to be careful to not copy from zeros due to shifting. 
            mask_tensor = utils.binary_mask_to_box_mask(mask_tensor, to_device=False)

            mask_tensor_expanded = mask_tensor[None, None, None, ...].to(dtype)
            composed_latents[0] = composed_latents[0] * (1. - mask_tensor_expanded) + latents_all[0] * mask_tensor_expanded
    
    # This is still needed with `compose_box_to_bg` to ensure the foreground latent is still visible and to compute foreground indices.
    for mask_idx in mask_order:
        latents_all, mask_tensor = latents_all_list[mask_idx], mask_tensor_list[mask_idx]
        foreground_indices = foreground_indices * (~mask_tensor) + (mask_idx + 1) * mask_tensor
        mask_tensor_expanded = mask_tensor[None, None, None, ...].to(dtype)
        composed_latents = composed_latents * (1. - mask_tensor_expanded) + latents_all * mask_tensor_expanded
        
    composed_latents, foreground_indices = composed_latents.to(torch_device), foreground_indices.to(torch_device)
    return composed_latents, foreground_indices

def align_with_bboxes(latents_all_list, mask_tensor_list, bboxes, horizontal_shift_only=False):
    """
    Each offset in `offset_list` is `(x_offset, y_offset)` (normalized).
    """
    new_latents_all_list, new_mask_tensor_list, offset_list = [], [], []
    for latents_all, mask_tensor, bbox in zip(latents_all_list, mask_tensor_list, bboxes):
        x_src_center, y_src_center = utils.binary_mask_to_center(mask_tensor, normalize=True)
        x_min_dest, y_min_dest, x_max_dest, y_max_dest = bbox
        x_dest_center, y_dest_center = (x_min_dest + x_max_dest) / 2, (y_min_dest + y_max_dest) / 2
        # print("src (x,y):", x_src_center, y_src_center, "dest (x,y):", x_dest_center, y_dest_center)
        x_offset, y_offset = x_dest_center - x_src_center, y_dest_center - y_src_center
        if horizontal_shift_only:
            y_offset = 0.
        offset = x_offset, y_offset
        latents_all = utils.shift_tensor(latents_all, x_offset, y_offset, offset_normalized=True)
        mask_tensor = utils.shift_tensor(mask_tensor, x_offset, y_offset, offset_normalized=True)
        new_latents_all_list.append(latents_all)
        new_mask_tensor_list.append(mask_tensor)
        offset_list.append(offset)

    return new_latents_all_list, new_mask_tensor_list, offset_list

@torch.no_grad()
def compose_latents_with_alignment(
    model_dict, latents_all_list, mask_tensor_list, num_inference_steps, overall_batch_size, height, width,
    align_with_overall_bboxes=True, overall_bboxes=None, horizontal_shift_only=False, **kwargs
):
    if align_with_overall_bboxes and len(latents_all_list):
        expanded_overall_bboxes = utils.expand_overall_bboxes(overall_bboxes)
        latents_all_list, mask_tensor_list, offset_list = align_with_bboxes(latents_all_list, mask_tensor_list, bboxes=expanded_overall_bboxes, horizontal_shift_only=horizontal_shift_only)
    else:
        offset_list = [(0., 0.) for _ in range(len(latents_all_list))]
    composed_latents, foreground_indices = compose_latents(model_dict, latents_all_list, mask_tensor_list, num_inference_steps, overall_batch_size, height, width, **kwargs)
    return composed_latents, foreground_indices, offset_list

def get_input_latents_list(model_dict, bg_seed, fg_seed_start, fg_blending_ratio, height, width, so_prompt_phrase_box_list=None, so_boxes=None, verbose=False):
    """
    Note: the returned input latents are scaled by `scheduler.init_noise_sigma`
    """
    unet, scheduler, dtype = model_dict.unet, model_dict.scheduler, model_dict.dtype
    
    generator_bg = torch.manual_seed(bg_seed)  # Seed generator to create the inital latent noise
    latents_bg = get_unscaled_latents(batch_size=1, in_channels=unet.config.in_channels, height=height, width=width, generator=generator_bg, dtype=dtype)

    input_latents_list = []
    
    if so_boxes is None:
        # For compatibility
        so_boxes = [item[-1] for item in so_prompt_phrase_box_list]
    
    # change this changes the foreground initial noise
    for idx, obj_box in enumerate(so_boxes):
        H, W = height // 8, width // 8
        fg_mask = utils.proportion_to_mask(obj_box, H, W)

        if verbose:
            plt.imshow(fg_mask.cpu().numpy())
            plt.show()
        
        fg_seed = fg_seed_start + idx
        if fg_seed == bg_seed:
            # We should have different seeds for foreground and background
            fg_seed += 12345
        
        generator_fg = torch.manual_seed(fg_seed)
        latents_fg = get_unscaled_latents(batch_size=1, in_channels=unet.config.in_channels, height=height, width=width, generator=generator_fg, dtype=dtype)
    
        input_latents = blend_latents(latents_bg, latents_fg, fg_mask, fg_blending_ratio=fg_blending_ratio)
    
        input_latents = input_latents * scheduler.init_noise_sigma
    
        input_latents_list.append(input_latents)
    
    latents_bg = latents_bg * scheduler.init_noise_sigma
    
    return input_latents_list, latents_bg

