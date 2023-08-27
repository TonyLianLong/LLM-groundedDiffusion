import torch
import torch.nn.functional as F
import math
from collections.abc import Iterable
import warnings

import utils

# A list mapping: prompt index to str (prompt in a list of token str)
def get_token_map(tokenizer, prompt, verbose=False, padding="do_not_pad"):
    fg_prompt_tokens = tokenizer([prompt], padding=padding, max_length=77, return_tensors="np")
    input_ids = fg_prompt_tokens['input_ids'][0]
    
    # index_to_last_with = np.max(np.where(input_ids == 593))
    # index_to_last_eot = np.max(np.where(input_ids == 49407))
    
    token_map = []
    for ind, item in enumerate(input_ids.tolist()):

        token = tokenizer._convert_id_to_token(item)
        if verbose:
            print(f"{ind}, {token} ({item})")
            
        token_map.append(token)
        
        # If we don't pad, we don't need to break.
        # if item == tokenizer.eos_token_id:
        #     break
    
    return token_map

def get_phrase_indices(tokenizer, prompt, phrases, verbose=False, words=None, include_eos=False, token_map=None, return_word_token_indices=False, add_suffix_if_not_found=False):
    for obj in phrases:
        # Suffix the prompt with object name for attention guidance if object is not in the prompt, using "|" to separate the prompt and the suffix
        if obj not in prompt:
            prompt += "| " + obj

    if token_map is None:
        # We allow using a pre-computed token map.
        token_map = get_token_map(tokenizer, prompt=prompt, verbose=verbose, padding="do_not_pad")
    token_map_str = " ".join(token_map)

    object_positions = []
    word_token_indices = []
    for obj_ind, obj in enumerate(phrases):
        phrase_token_map = get_token_map(tokenizer, prompt=obj, verbose=verbose, padding="do_not_pad")
        # Remove <bos> and <eos> in substr
        phrase_token_map = phrase_token_map[1:-1]
        phrase_token_map_len = len(phrase_token_map)
        phrase_token_map_str = " ".join(phrase_token_map)
        
        if verbose:
            print("Full str:", token_map_str, "Substr:", phrase_token_map_str, "Phrase:", phrases)
        
        # Count the number of token before substr
        # The substring comes with a trailing space that needs to be removed by minus one in the index.
        obj_first_index = len(token_map_str[:token_map_str.index(phrase_token_map_str)-1].split(" "))

        obj_position = list(range(obj_first_index, obj_first_index + phrase_token_map_len))
        if include_eos:
            obj_position.append(token_map.index(tokenizer.eos_token))
        object_positions.append(obj_position)
        
        if return_word_token_indices:
            # Picking the last token in the specification
            if words is None:
                so_token_index = object_positions[0][-1]
                # Picking the noun or perform pooling on attention with the tokens may be better
                print(f"Picking the last token \"{token_map[so_token_index]}\" ({so_token_index}) as attention token for extracting attention for SAM, which might not be the right one")
            else:
                word = words[obj_ind]
                word_token_map = get_token_map(tokenizer, prompt=word, verbose=verbose, padding="do_not_pad")
                # Get the index of the last token of word (the occurrence in phrase) in the prompt. Note that we skip the <eos> token through indexing with -2.
                so_token_index = obj_first_index + phrase_token_map.index(word_token_map[-2])

            if verbose:
                print("so_token_index:", so_token_index)
            
            word_token_indices.append(so_token_index)

    if return_word_token_indices:
        if add_suffix_if_not_found:
            return object_positions, word_token_indices, prompt
        return object_positions, word_token_indices

    if add_suffix_if_not_found:
        return object_positions, prompt

    return object_positions

def add_ca_loss_per_attn_map_to_loss(loss, attn_map, object_number, bboxes, object_positions, use_ratio_based_loss=True, fg_top_p=0.2, bg_top_p=0.2, fg_weight=1.0, bg_weight=1.0, verbose=False):
    """
    fg_top_p, bg_top_p, fg_weight, and bg_weight are only used with max-based loss
    """
    
    # Uncomment to debug:
    # print(fg_top_p, bg_top_p, fg_weight, bg_weight)
    
    # b is the number of heads, not batch
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))
    for obj_idx in range(object_number):
        obj_loss = 0
        mask = torch.zeros(size=(H, W), device="cuda")
        obj_boxes = bboxes[obj_idx]

        # We support two level (one box per phrase) and three level (multiple boxes per phrase)
        if not isinstance(obj_boxes[0], Iterable):
            obj_boxes = [obj_boxes]

        for obj_box in obj_boxes:
            # x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
            x_min, y_min, x_max, y_max = utils.scale_proportion(obj_box, H=H, W=W)
            mask[y_min: y_max, x_min: x_max] = 1

        for obj_position in object_positions[obj_idx]:
            # Could potentially optimize to compute this for loop in batch.
            # Could crop the ref cross attention before saving to save memory.
            
            ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)

            if use_ratio_based_loss:
                warnings.warn("Using ratio-based loss, which is deprecated. Max-based loss is recommended. The scale may be different.")
                # Original loss function (ratio-based loss function)
                
                # Enforces the attention to be within the mask only. Does not enforce within-mask distribution.
                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                obj_loss += torch.mean((1 - activation_value) ** 2)
                # if verbose:
                #     print(f"enforce attn to be within the mask loss: {torch.mean((1 - activation_value) ** 2).item():.2f}")
            else:
                # Max-based loss function
                
                # shape: (b, H * W)
                ca_map_obj = attn_map[:, :, obj_position] # .reshape(b, H, W)
                k_fg = (mask.sum() * fg_top_p).long().clamp_(min=1)
                k_bg = ((1 - mask).sum() * bg_top_p).long().clamp_(min=1)
                
                mask_1d = mask.view(1, -1)
                
                # Take the topk over spatial dimension, and then take the sum over heads dim
                # The mean is over k_fg and k_bg dimension, so we don't need to sum and divide on our own.
                obj_loss += (1 - (ca_map_obj * mask_1d).topk(k=k_fg).values.mean(dim=1)).sum(dim=0) * fg_weight
                obj_loss += ((ca_map_obj * (1 - mask_1d)).topk(k=k_bg).values.mean(dim=1)).sum(dim=0) * bg_weight    

        loss += obj_loss / len(object_positions[obj_idx])
        
    return loss

def add_ref_ca_loss_per_attn_map_to_lossv2(loss, saved_attn, object_number, bboxes, object_positions, guidance_attn_keys, ref_ca_saved_attns, ref_ca_last_token_only, ref_ca_word_token_only, word_token_indices, index, loss_weight, eps=1e-5, verbose=False):
    """
    This adds the ca loss with ref. Note that this should be used with ca loss without ref since it only enforces the mse of the normalized ca between ref and target.
    
    `ref_ca_saved_attn` should have the same structure as bboxes and object_positions (until the inner content, which should be similar to saved_attn).
    """
    
    if loss_weight == 0.:
        # Skip computing the reference loss if the loss weight is 0.
        return loss

    for obj_idx in range(object_number):
        obj_loss = 0
        
        obj_boxes = bboxes[obj_idx]
        obj_ref_ca_saved_attns = ref_ca_saved_attns[obj_idx]

        # We support two level (one box per phrase) and three level (multiple boxes per phrase)
        if not isinstance(obj_boxes[0], Iterable):
            obj_boxes = [obj_boxes]
            obj_ref_ca_saved_attns = [obj_ref_ca_saved_attns]

        assert len(obj_boxes) == len(obj_ref_ca_saved_attns), f"obj_boxes: {len(obj_boxes)}, obj_ref_ca_saved_attns: {len(obj_ref_ca_saved_attns)}"

        for obj_box, obj_ref_ca_saved_attn in zip(obj_boxes, obj_ref_ca_saved_attns):
            # obj_ref_ca_map_items has all timesteps.
            # Format: (timestep (index), attn_key, batch, heads, 2d dim, num text tokens (selected 1))
            
            # Different from ca_loss without ref, which has one loss for all boxes for a phrase (a set of object positions), we have one loss per box.
            
            # obj_ref_ca_saved_attn_items: select the timestep
            obj_ref_ca_saved_attn = obj_ref_ca_saved_attn[index]
            
            for attn_key in guidance_attn_keys:
                attn_map = saved_attn[attn_key]
                if not attn_map.is_cuda:
                    attn_map = attn_map.cuda()
                attn_map = attn_map.squeeze(dim=0)
                
                obj_ref_ca_map = obj_ref_ca_saved_attn[attn_key]
                if not obj_ref_ca_map.is_cuda:
                    obj_ref_ca_map = obj_ref_ca_map.cuda()
                # obj_ref_ca_map: (batch, heads, 2d dim, num text token)
                # `squeeze` on `obj_ref_ca_map` is combined with the subsequent indexing
                
                # b is the number of heads, not batch
                b, i, j = attn_map.shape
                H = W = int(math.sqrt(i))
                # `obj_ref_ca_map` only has one text token (the 0 at the last dimension)
                
                assert obj_ref_ca_map.ndim == 4, f"{obj_ref_ca_map.shape}"
                obj_ref_ca_map = obj_ref_ca_map[0, :, :, 0]

                # Same mask for all heads
                obj_mask = torch.zeros(size=(H, W), device="cuda")
                # x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                x_min, y_min, x_max, y_max = utils.scale_proportion(obj_box, H=H, W=W)
                obj_mask[y_min: y_max, x_min: x_max] = 1
                
                # keep 1d mask
                obj_mask = obj_mask.reshape(1, -1)

                # Optimize the loss over the last phrase token only (assuming the indices in `object_positions[obj_idx]` is sorted)
                if ref_ca_word_token_only:
                    object_positions_to_iterate = [word_token_indices[obj_idx]]
                elif ref_ca_last_token_only:
                    object_positions_to_iterate = [object_positions[obj_idx][-1]]
                else:
                    print(f"Applying attention transfer from one attention to all attention maps in object positions {object_positions[obj_idx]}, which is likely to be incorrect")
                    object_positions_to_iterate = object_positions[obj_idx]
                for obj_position in object_positions_to_iterate:
                    ca_map_obj = attn_map[:, :, obj_position]
                    
                    ca_map_obj_masked = ca_map_obj * obj_mask
                    
                    # Add eps because the sum can be very small, causing NaN
                    ca_map_obj_masked_normalized = ca_map_obj_masked / (ca_map_obj_masked.sum(dim=-1, keepdim=True) + eps)
                    obj_ref_ca_map_masked = obj_ref_ca_map * obj_mask
                    obj_ref_ca_map_masked_normalized = obj_ref_ca_map_masked / (obj_ref_ca_map_masked.sum(dim=-1, keepdim=True) + eps)
                    
                    # We found dividing by object mask size makes the loss too small. Since the normalized masked attn has mean value inversely proportional to the mask size, summing the values up spatially gives a relatively standard scale to add to other losses.
                    activation_value = (torch.abs(ca_map_obj_masked_normalized - obj_ref_ca_map_masked_normalized)).sum(dim=-1)

                    obj_loss += torch.mean(activation_value, dim=0)
        
        # The normalization for len(obj_ref_ca_map_items) is at the outside of this function.
        # Note that we assume we have at least one box for each object
        loss += loss_weight * obj_loss / (len(obj_boxes) * len(object_positions_to_iterate))
    
        if verbose:
            print(f"reference cross-attention obj_loss: unweighted {obj_loss.item() / (len(obj_boxes) * len(object_positions[obj_idx])):.3f}, weighted {loss_weight * obj_loss.item() / (len(obj_boxes) * len(object_positions[obj_idx])):.3f}")
        
    return loss

def compute_ca_lossv3(saved_attn, bboxes, object_positions, guidance_attn_keys, ref_ca_saved_attns=None, ref_ca_last_token_only=True, ref_ca_word_token_only=False, word_token_indices=None, index=None, ref_ca_loss_weight=1.0, verbose=False, **kwargs):
    """
    The `saved_attn` is supposed to be passed to `save_attn_to_dict` in `cross_attention_kwargs` prior to computing ths loss.
    `AttnProcessor` will put attention maps into the `save_attn_to_dict`.
    
    `index` is the timestep.
    `ref_ca_word_token_only`: This has precedence over `ref_ca_last_token_only` (i.e., if both are enabled, we take the token from word rather than the last token).
    `ref_ca_last_token_only`: `ref_ca_saved_attn` comes from the attention map of the last token of the phrase in single object generation, so we apply it only to the last token of the phrase in overall generation if this is set to True. If set to False, `ref_ca_saved_attn` will be applied to all the text tokens.
    """
    loss = torch.tensor(0).float().cuda()
    object_number = len(bboxes)
    if object_number == 0:
        return loss
    
    for attn_key in guidance_attn_keys:
        # We only have 1 cross attention for mid.
        attn_map_integrated = saved_attn[attn_key]
        if not attn_map_integrated.is_cuda:
            attn_map_integrated = attn_map_integrated.cuda()
        # Example dimension: [20, 64, 77]
        attn_map = attn_map_integrated.squeeze(dim=0)
        loss = add_ca_loss_per_attn_map_to_loss(loss, attn_map, object_number, bboxes, object_positions, verbose=verbose, **kwargs)    

    num_attn = len(guidance_attn_keys)

    if num_attn > 0:
        loss = loss / (object_number * num_attn)

    if ref_ca_saved_attns is not None:
        ref_loss = torch.tensor(0).float().cuda()
        ref_loss = add_ref_ca_loss_per_attn_map_to_lossv2(
            ref_loss, saved_attn=saved_attn, object_number=object_number, bboxes=bboxes, object_positions=object_positions, guidance_attn_keys=guidance_attn_keys,
            ref_ca_saved_attns=ref_ca_saved_attns, ref_ca_last_token_only=ref_ca_last_token_only, ref_ca_word_token_only=ref_ca_word_token_only, word_token_indices=word_token_indices, verbose=verbose, index=index, loss_weight=ref_ca_loss_weight
        )
        
        num_attn = len(guidance_attn_keys)
        
        if verbose:
            print(f"loss {loss.item():.3f}, reference attention loss (weighted) {ref_loss.item() / (object_number * num_attn):.3f}")
        
        loss += ref_loss / (object_number * num_attn)
    
    return loss
