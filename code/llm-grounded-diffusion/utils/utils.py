import torch
from PIL import ImageDraw
import numpy as np
import os
import gc

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def draw_box(pil_img, bboxes, phrases):
    draw = ImageDraw.Draw(pil_img)
    # font = ImageFont.truetype('./FreeMono.ttf', 25)

    for obj_bbox, phrase in zip(bboxes, phrases):
        x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
        draw.rectangle([int(x_0 * 512), int(y_0 * 512), int(x_1 * 512), int(y_1 * 512)], outline='red', width=5)
        draw.text((int(x_0 * 512) + 5, int(y_0 * 512) + 5), phrase, font=None, fill=(255, 0, 0))
    
    return pil_img

def get_centered_box(box, horizontal_center_only=True):
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    
    if horizontal_center_only:
        return [0.5 - w/2, y_min, 0.5 + w/2, y_max]
    
    h = y_max - y_min
    
    return [0.5 - w/2, 0.5 - h/2, 0.5 + w/2, 0.5 + h/2]

# NOTE: this changes the behavior of the function
def proportion_to_mask(obj_box, H, W, use_legacy=False, return_np=False):
    x_min, y_min, x_max, y_max = scale_proportion(obj_box, H, W, use_legacy)
    if return_np:
        mask = np.zeros((H, W))
    else:
        mask = torch.zeros(H, W).to(torch_device)
    mask[y_min: y_max, x_min: x_max] = 1.

    return mask

def scale_proportion(obj_box, H, W, use_legacy=False):
    if use_legacy:
        # Bias towards the top-left corner
        x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
    else:
        # Separately rounding box_w and box_h to allow shift invariant box sizes. Otherwise box sizes may change when both coordinates being rounded end with ".5".
        x_min, y_min = round(obj_box[0] * W), round(obj_box[1] * H)
        box_w, box_h = round((obj_box[2] - obj_box[0]) * W), round((obj_box[3] - obj_box[1]) * H)
        x_max, y_max = x_min + box_w, y_min + box_h
        
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, W), min(y_max, H)
        
    return x_min, y_min, x_max, y_max

def binary_mask_to_box(mask, enlarge_box_by_one=True, w_scale=1, h_scale=1):
    if isinstance(mask, torch.Tensor):
        mask_loc = torch.where(mask)
    else:
        mask_loc = np.where(mask)
    height, width = mask.shape
    if len(mask_loc) == 0:
        raise ValueError('The mask is empty')
    if enlarge_box_by_one:
        ymin, ymax = max(min(mask_loc[0]) - 1, 0), min(max(mask_loc[0]) + 1, height)
        xmin, xmax = max(min(mask_loc[1]) - 1, 0), min(max(mask_loc[1]) + 1, width)
    else:
        ymin, ymax = min(mask_loc[0]), max(mask_loc[0])
        xmin, xmax = min(mask_loc[1]), max(mask_loc[1])
    box = [xmin * w_scale, ymin * h_scale, xmax * w_scale, ymax * h_scale]

    return box

def binary_mask_to_box_mask(mask, to_device=True):
    box = binary_mask_to_box(mask)
    x_min, y_min, x_max, y_max = box
    
    H, W = mask.shape
    mask = torch.zeros(H, W)
    if to_device:
        mask = mask.to(torch_device)
    mask[y_min: y_max+1, x_min: x_max+1] = 1.
    
    return mask

def binary_mask_to_center(mask, normalize=False):
    """
    This computes the mass center of the mask.
    normalize: the coords range from 0 to 1
    
    Reference: https://stackoverflow.com/a/66184125
    """
    h, w = mask.shape
    
    total = mask.sum()
    if isinstance(mask, torch.Tensor):
        x_coord = ((mask.sum(dim=0) @ torch.arange(w)) / total).item()
        y_coord = ((mask.sum(dim=1) @ torch.arange(h)) / total).item()
    else:
        x_coord = (mask.sum(axis=0) @ np.arange(w)) / total
        y_coord = (mask.sum(axis=1) @ np.arange(h)) / total
    
    if normalize:
        x_coord, y_coord = x_coord / w, y_coord / h
    return x_coord, y_coord
    

def iou(mask, masks, eps=1e-6):
    # mask: [h, w], masks: [n, h, w]
    mask = mask[None].astype(bool)
    masks = masks.astype(bool)
    i = (mask & masks).sum(axis=(1,2))
    u = (mask | masks).sum(axis=(1,2))
    
    return i / (u + eps)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def expand_overall_bboxes(overall_bboxes):
    """
    Expand overall bboxes from a 3d list to 2d list:
    Input: [[box 1 for phrase 1, box 2 for phrase 1], ...]
    Output: [box 1, box 2, ...]
    """
    return sum(overall_bboxes, start=[])

def shift_tensor(tensor, x_offset, y_offset, base_w=8, base_h=8, offset_normalized=False, ignore_last_dim=False):
    """base_w and base_h: make sure the shift is aligned in the latent and multiple levels of cross attention"""
    if ignore_last_dim:
        tensor_h, tensor_w = tensor.shape[-3:-1]
    else:
        tensor_h, tensor_w = tensor.shape[-2:]
    if offset_normalized:
        assert tensor_h % base_h == 0 and tensor_w % base_w == 0, f"{tensor_h, tensor_w} is not a multiple of {base_h, base_w}"
        scale_from_base_h, scale_from_base_w = tensor_h // base_h, tensor_w // base_w
        x_offset, y_offset = round(x_offset * base_w) * scale_from_base_w, round(y_offset * base_h) * scale_from_base_h
    new_tensor = torch.zeros_like(tensor)
    
    overlap_w = tensor_w - abs(x_offset)
    overlap_h = tensor_h - abs(y_offset)
    
    if y_offset >= 0:
        y_src_start = 0
        y_dest_start = y_offset
    else:
        y_src_start = -y_offset
        y_dest_start = 0
    
    if x_offset >= 0:
        x_src_start = 0
        x_dest_start = x_offset
    else:
        x_src_start = -x_offset
        x_dest_start = 0
    
    if ignore_last_dim:
        # For cross attention maps, the third to last and the second to last are the 2D dimensions after unflatten.
        new_tensor[..., y_dest_start:y_dest_start+overlap_h, x_dest_start:x_dest_start+overlap_w, :] = tensor[..., y_src_start:y_src_start+overlap_h, x_src_start:x_src_start+overlap_w, :]
    else:
        new_tensor[..., y_dest_start:y_dest_start+overlap_h, x_dest_start:x_dest_start+overlap_w] = tensor[..., y_src_start:y_src_start+overlap_h, x_src_start:x_src_start+overlap_w]

    return new_tensor
