"""
This is an reimplementation boxdiff baseline for reference and comparison. It is not used in the Web UI and not enabled by default since the current attention guidance implementation (in `guidance`), which uses attention maps from multiple levels and attention transfer, seems to be more robust and coherent.

Credit: https://github.com/showlab/BoxDiff/blob/master/pipeline/sd_pipeline_boxdiff.py
"""

import torch
import torch.nn.functional as F
import math
import warnings
import gc
from collections.abc import Iterable
import utils
from . import guidance
from .attn import GaussianSmoothing

from typing import Any, Callable, Dict, List, Optional, Union, Tuple


def _compute_max_attention_per_index(attention_maps: torch.Tensor,
                                     object_positions: List[List[int]],
                                     smooth_attentions: bool = False,
                                     sigma: float = 0.5,
                                     kernel_size: int = 3,
                                     normalize_eot: bool = False,
                                     bboxes: List[List[int]] = None,
                                     P: float = 0.2,
                                     L: int = 1,
                                     ) -> List[torch.Tensor]:
    """ Computes the maximum attention value for each of the tokens we wish to alter. """
    last_idx = -1
    assert not normalize_eot, "normalize_eot is unimplemented"

    attention_for_text = attention_maps[:, :, 1:last_idx]
    attention_for_text *= 100
    attention_for_text = F.softmax(attention_for_text, dim=-1)

    # Extract the maximum values
    max_indices_list_fg = []
    max_indices_list_bg = []
    dist_x = []
    dist_y = []

    for obj_idx, text_positions_per_obj in enumerate(object_positions):
        for text_position_per_obj in text_positions_per_obj:
            # Shift indices since we removed the first token
            image = attention_for_text[:, :, text_position_per_obj - 1]
            H, W = image.shape

            obj_mask = torch.zeros_like(image)
            corner_mask_x = torch.zeros(
                (W,), device=obj_mask.device, dtype=obj_mask.dtype)
            corner_mask_y = torch.zeros(
                (H,), device=obj_mask.device, dtype=obj_mask.dtype)

            obj_boxes = bboxes[obj_idx]

            # We support two level (one box per phrase) and three level (multiple boxes per phrase)
            if not isinstance(obj_boxes[0], Iterable):
                obj_boxes = [obj_boxes]

            for obj_box in obj_boxes:
                x_min, y_min, x_max, y_max = utils.scale_proportion(
                    obj_box, H=H, W=W)
                obj_mask[y_min: y_max, x_min: x_max] = 1

                corner_mask_x[max(x_min - L, 0): min(x_min + L + 1, W)] = 1.
                corner_mask_x[max(x_max - L, 0): min(x_max + L + 1, W)] = 1.
                corner_mask_y[max(y_min - L, 0): min(y_min + L + 1, H)] = 1.
                corner_mask_y[max(y_max - L, 0): min(y_max + L + 1, H)] = 1.

            bg_mask = 1 - obj_mask

            if smooth_attentions:
                smoothing = GaussianSmoothing(
                    channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0),
                              (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)

            # Inner-Box constraint
            k = (obj_mask.sum() * P).long()
            max_indices_list_fg.append(
                (image * obj_mask).reshape(-1).topk(k)[0].mean())

            # Outer-Box constraint
            k = (bg_mask.sum() * P).long()
            max_indices_list_bg.append(
                (image * bg_mask).reshape(-1).topk(k)[0].mean())

            # Corner Constraint
            gt_proj_x = torch.max(obj_mask, dim=0).values
            gt_proj_y = torch.max(obj_mask, dim=1).values

            # create gt according to the number L
            dist_x.append((F.l1_loss(image.max(dim=0)[
                          0], gt_proj_x, reduction='none') * corner_mask_x).mean())
            dist_y.append((F.l1_loss(image.max(dim=1)[
                          0], gt_proj_y, reduction='none') * corner_mask_y).mean())

    return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y


def _compute_loss(max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                  dist_x: List[torch.Tensor], dist_y: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
    """ Computes the attend-and-excite loss using the maximum attention value for each token. """
    losses_fg = [max(0, 1. - curr_max)
                 for curr_max in max_attention_per_index_fg]
    losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
    loss = sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y)

    # print(f"{losses_fg}, {losses_bg}, {dist_x}, {dist_y}, {loss}")

    if return_losses:
        return max(losses_fg), losses_fg
    else:
        return max(losses_fg), loss


def compute_ca_loss_boxdiff(saved_attn, bboxes, object_positions, guidance_attn_keys, ref_ca_saved_attns=None, ref_ca_last_token_only=True, ref_ca_word_token_only=False, word_token_indices=None, index=None, ref_ca_loss_weight=1.0, verbose=False, **kwargs):
    """
    v3 is equivalent to v2 but with new dictionary format for attention maps.
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

    attn_map_list = []

    for attn_key in guidance_attn_keys:
        # We only have 1 cross attention for mid.
        attn_map_integrated = saved_attn[attn_key]
        if not attn_map_integrated.is_cuda:
            attn_map_integrated = attn_map_integrated.cuda()
        # Example dimension: [20, 64, 77]
        attn_map = attn_map_integrated.squeeze(dim=0)
        attn_map_list.append(attn_map)
    # This averages both across layers and across attention heads
    attn_map = torch.cat(attn_map_list, dim=0).mean(dim=0)
    loss = add_ca_loss_per_attn_map_to_loss_boxdiff(
        loss, attn_map, object_number, bboxes, object_positions, verbose=verbose, **kwargs)

    if ref_ca_saved_attns is not None:
        warnings.warn('Attention reference loss is enabled in boxdiff mode. The original boxdiff does not have attention reference loss.')
        
        ref_loss = torch.tensor(0).float().cuda()
        ref_loss = guidance.add_ref_ca_loss_per_attn_map_to_lossv2(
            ref_loss, saved_attn=saved_attn, object_number=object_number, bboxes=bboxes, object_positions=object_positions, guidance_attn_keys=guidance_attn_keys,
            ref_ca_saved_attns=ref_ca_saved_attns, ref_ca_last_token_only=ref_ca_last_token_only, ref_ca_word_token_only=ref_ca_word_token_only, word_token_indices=word_token_indices, verbose=verbose, index=index, loss_weight=ref_ca_loss_weight
        )
        print(f"loss {loss.item():.3f}, reference attention loss (weighted) {ref_loss.item():.3f}")
        loss += ref_loss

    return loss


def add_ca_loss_per_attn_map_to_loss_boxdiff(original_loss, attention_maps, object_number, bboxes, object_positions, P=0.2, L=1, smooth_attentions=True, sigma=0.5, kernel_size=3, normalize_eot=False, verbose=False):
    # NOTE: normalize_eot is enabled in SD v2.1 in boxdiff
    i, j = attention_maps.shape
    H = W = int(math.sqrt(i))

    attention_maps = attention_maps.view(H, W, j)
    # attention_maps is aggregated cross attn map across layers and steps
    # attention_maps shape: [H, W, 77]
    max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = _compute_max_attention_per_index(
        attention_maps=attention_maps,
        object_positions=object_positions,
        smooth_attentions=smooth_attentions,
        sigma=sigma,
        kernel_size=kernel_size,
        normalize_eot=normalize_eot,
        bboxes=bboxes,
        P=P,
        L=L
    )

    _, loss = _compute_loss(max_attention_per_index_fg,
                            max_attention_per_index_bg, dist_x, dist_y)

    return original_loss + loss


def latent_backward_guidance_boxdiff(scheduler, unet, cond_embeddings, index, bboxes, object_positions, t, latents, loss, amp_loss_scale=10, latent_scale=20, scale_range=(1., 0.5), max_index_step=25, cross_attention_kwargs=None, ref_ca_saved_attns=None, guidance_attn_keys=None, verbose=False, **kwargs):
    """
    amp_loss_scale: this scales the loss but will de-scale before applying for latents. This is to prevent overflow/underflow with amp, not to adjust the update step size.
    latent_scale: this scales the step size for update (scale_factor in boxdiff).
    """

    if index < max_index_step:
        saved_attn = {}
        full_cross_attention_kwargs = {
            'save_attn_to_dict': saved_attn,
            'save_keys': guidance_attn_keys,
        }

        if cross_attention_kwargs is not None:
            full_cross_attention_kwargs.update(cross_attention_kwargs)

        latents.requires_grad_(True)
        latent_model_input = latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        unet(latent_model_input, t, encoder_hidden_states=cond_embeddings,
             return_cross_attention_probs=False, cross_attention_kwargs=full_cross_attention_kwargs)

        # TODO: could return the attention maps for the required blocks only and not necessarily the final output
        # update latents with guidance
        loss = compute_ca_loss_boxdiff(saved_attn=saved_attn, bboxes=bboxes, object_positions=object_positions, guidance_attn_keys=guidance_attn_keys,
                                       ref_ca_saved_attns=ref_ca_saved_attns, index=index, verbose=verbose, **kwargs) * amp_loss_scale

        if torch.isnan(loss):
            print("**Loss is NaN**")

        del full_cross_attention_kwargs, saved_attn
        # call gc.collect() here may release some memory

        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents])[0]

        latents.requires_grad_(False)

        if True:
            warnings.warn("Using guidance scaled with sqrt scale")
            # According to boxdiff's implementation: https://github.com/Sierkinhane/BoxDiff/blob/16ffb677a9128128e04553a0200870a526731be0/pipeline/sd_pipeline_boxdiff.py#L616
            scale = (scale_range[0] + (scale_range[1] - scale_range[0])
                     * index / (len(scheduler.timesteps) - 1)) ** (0.5)
            latents = latents - latent_scale * scale / amp_loss_scale * grad_cond
        elif hasattr(scheduler, 'sigmas'):
            warnings.warn("Using guidance scaled with sigmas")
            scale = scheduler.sigmas[index] ** 2
            latents = latents - grad_cond * scale
        elif hasattr(scheduler, 'alphas_cumprod'):
            warnings.warn("Using guidance scaled with alphas_cumprod")
            # Scaling with classifier guidance
            alpha_prod_t = scheduler.alphas_cumprod[t]
            # Classifier guidance: https://arxiv.org/pdf/2105.05233.pdf
            # DDIM: https://arxiv.org/pdf/2010.02502.pdf
            scale = (1 - alpha_prod_t) ** (0.5)
            latents = latents - latent_scale * scale / amp_loss_scale * grad_cond
        else:
            warnings.warn("No scaling in guidance is performed")
            scale = 1
            latents = latents - grad_cond

        gc.collect()
        torch.cuda.empty_cache()

        if verbose:
            print(
                f"time index {index}, loss: {loss.item() / amp_loss_scale:.3f} (de-scaled with scale {amp_loss_scale:.1f}), latent grad scale: {scale:.3f}")

    return latents, loss
