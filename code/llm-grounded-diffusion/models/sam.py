import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from models import torch_device
from transformers import SamModel, SamProcessor
import utils
import cv2
from scipy import ndimage

def load_sam():
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(torch_device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    sam_model_dict = dict(
        sam_model = sam_model, sam_processor = sam_processor
    )

    return sam_model_dict

# Not fully backward compatible with the previous implementation
# Reference: lmdv2/notebooks/gen_masked_latents_multi_object_ref_ca_loss_modular.ipynb
def sam(sam_model_dict, image, input_points=None, input_boxes=None, target_mask_shape=None, return_numpy=True):
    """target_mask_shape: (h, w)"""
    sam_model, sam_processor = sam_model_dict['sam_model'], sam_model_dict['sam_processor']
    
    if input_boxes and isinstance(input_boxes[0], tuple):
        # Convert tuple to list
        input_boxes = [list(input_box) for input_box in input_boxes]
        
    if input_boxes and input_boxes[0] and isinstance(input_boxes[0][0], tuple):
        # Convert tuple to list
        input_boxes = [[list(input_box) for input_box in input_boxes_item] for input_boxes_item in input_boxes]
    
    with torch.no_grad():
        with torch.autocast(torch_device):
            inputs = sam_processor(image, input_points=input_points, input_boxes=input_boxes, return_tensors="pt").to(torch_device)
            outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu().float(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        conf_scores = outputs.iou_scores.cpu().numpy()[0,0]
        del inputs, outputs
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if return_numpy:
        masks = [F.interpolate(masks_item.type(torch.float), target_mask_shape, mode='bilinear').type(torch.bool).numpy() for masks_item in masks]
    else:
        masks = [F.interpolate(masks_item.type(torch.float), target_mask_shape, mode='bilinear').type(torch.bool) for masks_item in masks]

    return masks, conf_scores

def sam_point_input(sam_model_dict, image, input_points, **kwargs):
    return sam(sam_model_dict, image, input_points=input_points, **kwargs)
    
def sam_box_input(sam_model_dict, image, input_boxes, **kwargs):
    return sam(sam_model_dict, image, input_boxes=input_boxes, **kwargs)

def get_iou_with_resize(mask, masks, masks_shape):
    masks = np.array([cv2.resize(mask.astype(np.uint8) * 255, masks_shape[::-1], cv2.INTER_LINEAR).astype(bool) for mask in masks])
    return utils.iou(mask, masks)

def select_mask(masks, conf_scores, coarse_ious=None, rule="largest_over_conf", discourage_mask_below_confidence=0.85, discourage_mask_below_coarse_iou=0.2, verbose=False):
    """masks: numpy bool array"""
    mask_sizes = masks.sum(axis=(1, 2))
    
    # Another possible rule: iou with the attention mask
    if rule == "largest_over_conf":
        # Use the largest segmentation
        # Discourage selecting masks with conf too low or coarse iou is too low
        max_mask_size = np.max(mask_sizes)
        if coarse_ious is not None:
            scores = mask_sizes - (conf_scores < discourage_mask_below_confidence) * max_mask_size - (coarse_ious < discourage_mask_below_coarse_iou) * max_mask_size
        else:
            scores = mask_sizes - (conf_scores < discourage_mask_below_confidence) * max_mask_size
        if verbose:
            print(f"mask_sizes: {mask_sizes}, scores: {scores}")
    else:
        raise ValueError(f"Unknown rule: {rule}")

    mask_id = np.argmax(scores)
    mask = masks[mask_id]
    
    selection_conf = conf_scores[mask_id]
    
    if coarse_ious is not None:
        selection_coarse_iou = coarse_ious[mask_id]
    else:
        selection_coarse_iou = None

    if verbose:
        # print(f"Confidences: {conf_scores}")
        print(f"Selected a mask with confidence: {selection_conf}, coarse_iou: {selection_coarse_iou}")

    if verbose:
        plt.figure(figsize=(10, 8))
        # plt.suptitle("After SAM")
        for ind in range(3):
            plt.subplot(1, 3, ind+1)
            # This is obtained before resize.
            plt.title(f"Mask {ind}, score {scores[ind]}, conf {conf_scores[ind]:.2f}, iou {coarse_ious[ind] if coarse_ious is not None else None:.2f}")
            plt.imshow(masks[ind])
        plt.tight_layout()
        plt.show()

    return mask, selection_conf

def preprocess_mask(token_attn_np_smooth, mask_th, n_erode_dilate_mask=0):
    token_attn_np_smooth_normalized = token_attn_np_smooth - token_attn_np_smooth.min()
    token_attn_np_smooth_normalized /= token_attn_np_smooth_normalized.max()
    mask_thresholded = token_attn_np_smooth_normalized > mask_th
    
    if n_erode_dilate_mask:
        mask_thresholded = ndimage.binary_erosion(mask_thresholded, iterations=n_erode_dilate_mask)
        mask_thresholded = ndimage.binary_dilation(mask_thresholded, iterations=n_erode_dilate_mask)
    
    return mask_thresholded

# The overall pipeline to refine the attention mask
def sam_refine_attn(sam_input_image, token_attn_np, model_dict, height, width, H, W, use_box_input, gaussian_sigma, mask_th_for_box, n_erode_dilate_mask_for_box, mask_th_for_point, discourage_mask_below_confidence, discourage_mask_below_coarse_iou, verbose):
    
    # token_attn_np is for visualizations
    token_attn_np_smooth = ndimage.gaussian_filter(token_attn_np, sigma=gaussian_sigma)

    # (w, h)
    mask_size_scale = height // token_attn_np_smooth.shape[1], width // token_attn_np_smooth.shape[0]

    if use_box_input:
        # box input
        mask_binary = preprocess_mask(token_attn_np_smooth, mask_th_for_box, n_erode_dilate_mask=n_erode_dilate_mask_for_box)

        input_boxes = utils.binary_mask_to_box(mask_binary, w_scale=mask_size_scale[0], h_scale=mask_size_scale[1])
        input_boxes = [input_boxes]

        masks, conf_scores = sam_box_input(model_dict, image=sam_input_image, input_boxes=input_boxes, target_mask_shape=(H, W))
    else:
        # point input
        mask_binary = preprocess_mask(token_attn_np_smooth, mask_th_for_point, n_erode_dilate_mask=0)

        # Uses the max coordinate only
        max_coord = np.unravel_index(token_attn_np_smooth.argmax(), token_attn_np_smooth.shape)
        # print("max_coord:", max_coord)
        input_points = [[[max_coord[1] * mask_size_scale[1], max_coord[0] * mask_size_scale[0]]]]

        masks, conf_scores = sam_point_input(model_dict, image=sam_input_image, input_points=input_points, target_mask_shape=(H, W))
        
    if verbose:
        plt.title("Coarse binary mask (for box for box input and for iou)")
        plt.imshow(mask_binary)
        plt.show()
    
    coarse_ious = get_iou_with_resize(mask_binary, masks, masks_shape=mask_binary.shape)

    mask_selected, conf_score_selected = select_mask(masks, conf_scores, coarse_ious=coarse_ious, 
                                                         rule="largest_over_conf", 
                                                         discourage_mask_below_confidence=discourage_mask_below_confidence, 
                                                         discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
                                                         verbose=True)

    return mask_selected, conf_score_selected

def sam_refine_box(sam_input_image, box, *args, **kwargs):
    sam_input_images, boxes = [sam_input_image], [box]
    return sam_refine_boxes(sam_input_images, boxes, *args, **kwargs)

def sam_refine_boxes(sam_input_images, boxes, model_dict, height, width, H, W, discourage_mask_below_confidence, discourage_mask_below_coarse_iou, verbose):
    # (w, h)
    input_boxes = [[utils.scale_proportion(box, H=height, W=width) for box in boxes_item] for boxes_item in boxes]

    masks, conf_scores = sam_box_input(model_dict, image=sam_input_images, input_boxes=input_boxes, target_mask_shape=(H, W))
    
    mask_selected_batched_list, conf_score_selected_batched_list = [], []
    
    for boxes_item, masks_item in zip(boxes, masks):
        mask_selected_list, conf_score_selected_list = [], []
        for box, three_masks in zip(boxes_item, masks_item):
            mask_binary = utils.proportion_to_mask(box, H, W, return_np=True)
            if verbose:
                # Also the box is the input for SAM
                plt.title("Binary mask from input box (for iou)")
                plt.imshow(mask_binary)
                plt.show()
                        
            coarse_ious = get_iou_with_resize(mask_binary, three_masks, masks_shape=mask_binary.shape)

            mask_selected, conf_score_selected = select_mask(three_masks, conf_scores, coarse_ious=coarse_ious, 
                                                                rule="largest_over_conf", 
                                                                discourage_mask_below_confidence=discourage_mask_below_confidence, 
                                                                discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
                                                                verbose=True)

            mask_selected_list.append(mask_selected)
            conf_score_selected_list.append(conf_score_selected)
        mask_selected_batched_list.append(mask_selected_list)
        conf_score_selected_batched_list.append(conf_score_selected_list)
    
    return mask_selected_batched_list, conf_score_selected_batched_list
