# This file uses code from `https://github.com/omerbt/MultiDiffusion`.
# The license of this file is the same as the MultiDiffusion code.


from utils import parse, latents
from utils.parse import size, filter_boxes, show_boxes, show_masks
from tqdm import tqdm
import gc
from PIL import Image
import numpy as np
import argparse
import torchvision.transforms as T
import torch.nn as nn
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
import models
from easydict import EasyDict

version = "multidiffusion"


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class MultiDiffusion(nn.Module):
    def __init__(self, device, sd_version="2.0", batch_size=2, hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f"[INFO] loading stable diffusion...")
        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.1_768":
            model_key = "stabilityai/stable-diffusion-2-1"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "2.0_768":
            model_key = "stabilityai/stable-diffusion-2"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            # For custom models or fine-tunes, allow people to use arbitrary versions
            model_key = self.sd_version
            # raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(
            self.device
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key, subfolder="text_encoder"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_key, subfolder="unet"
        ).to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        # self.unet = torch.compile(self.unet)

        self.batch_size = batch_size

        print(f"[INFO] loaded stable diffusion!")

    @torch.no_grad()
    def unet_batch(self, latent_model_input, t, encoder_hidden_states):
        latent_model_inputs = torch.split(latent_model_input, self.batch_size, dim=0)

        encoder_hidden_states_all = torch.split(
            encoder_hidden_states, self.batch_size, dim=0
        )

        noise_preds = []
        for latent_model_input, encoder_hidden_states in zip(
            latent_model_inputs, encoder_hidden_states_all
        ):
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states)[
                "sample"
            ]
            noise_preds.append(noise_pred)

        # print(f"noise_pred.shape: {noise_pred.shape}")
        return torch.cat(noise_preds, dim=0)

    @torch.no_grad()
    def get_random_background(self, n_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[
            :, :, None, None
        ].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def generate(
        self,
        masks,
        prompts,
        negative_prompts="",
        height=512,
        width=2048,
        num_inference_steps=50,
        guidance_scale=7.5,
        bootstrapping=20,
        indep_uncond=False,
        normalization=True,
        seed=None,
    ):
        if bootstrapping:
            # get bootstrapping backgrounds
            # can move this outside of the function to speed up generation. i.e., calculate in init
            bootstrapping_backgrounds = self.get_random_background(bootstrapping)
            print(f"With bootstrapping ({bootstrapping} steps)")
        else:
            bootstrapping_backgrounds = None
            print("No bootstrapping")

        print(prompts)

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(
            prompts, negative_prompts
        )  # [2 * len(prompts), 77, 768]

        # Define panorama grid and get views
        generator = torch.manual_seed(seed)
        latent = latents.get_unscaled_latents(
            batch_size=1,
            in_channels=self.unet.config.in_channels,
            height=height,
            width=width,
            generator=generator,
            dtype=masks.dtype,
        )
        if bootstrapping:
            noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast("cuda"):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                count.zero_()
                value.zero_()

                for h_start, h_end, w_start, w_end in views:
                    masks_view = masks[:, :, h_start:h_end, w_start:w_end]
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(
                        len(prompts), 1, 1, 1
                    )
                    masks_view_binary = (masks_view >= 0.5).type(masks_view.dtype)
                    if i < bootstrapping:
                        bg = bootstrapping_backgrounds[
                            torch.randint(0, bootstrapping, (len(prompts) - 1,))
                        ]
                        bg = self.scheduler.add_noise(
                            bg, noise[:, :, h_start:h_end, w_start:w_end], t
                        )

                        current_mask = (masks_view_binary[1:]).clamp_(0.0, 1.0)
                        latent_view[1:] = latent_view[1:] * current_mask + bg * (
                            1 - current_mask
                        )

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latent_view] * 2)

                    # predict the noise residual
                    # print(latent_model_input.shape)
                    # print(t.shape)
                    # print(text_embeds.shape)
                    if latent_model_input.shape[0] > self.batch_size:
                        noise_pred = self.unet_batch(
                            latent_model_input, t, encoder_hidden_states=text_embeds
                        )
                    else:
                        noise_pred = self.unet(
                            latent_model_input, t, encoder_hidden_states=text_embeds
                        )["sample"]

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if indep_uncond:
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    else:
                        # Use the same uncond but different directions
                        noise_pred = guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                        noise_pred[:] += noise_pred_uncond[:1]

                    # compute the denoising step with the reference model
                    latents_view_denoised = self.scheduler.step(
                        noise_pred, t, latent_view
                    )["prev_sample"]

                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latents_view_denoised * masks_view
                    ).sum(dim=0, keepdims=True)

                    if normalization:
                        count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(
                            dim=0, keepdims=True
                        )
                    else:
                        # No normalizations
                        count[:] = 1.0

                # take the MultiDiffusion step
                latent = torch.where(count > 0, value / count, value)

        # Img latents -> imgs
        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        return img


def preprocess_mask(mask_path, h, w, device):
    if isinstance(mask_path, str):
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = mask_path.astype(np.float32)
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
    return mask



def boxes_to_masks_prompts(boxes, fg_negative_prompt, first_top=True):
    # Processes boxes together: a pixel belongs to the first box in order
    h, w = size
    masks = []
    prompts = []
    if first_top:
        # first top means fill in reversely
        boxes = boxes[::-1]

    inds_arr = np.full((h, w), fill_value=-1, dtype=np.int32)
    for ind, box in enumerate(boxes):
        name, [bbox_x, bbox_y, bbox_w, bbox_h] = box["name"], box["bounding_box"]

        inds_arr[bbox_y : bbox_y + bbox_h, bbox_x : bbox_x + bbox_w] = ind
        prompt = f"{name}"
        prompts.append(prompt)

    for ind, box in enumerate(boxes):
        mask = (inds_arr == ind).astype(np.float32)

        masks.append(mask)

    fg_negative_prompts = [fg_negative_prompt] * len(masks)

    if first_top:
        # keep the order for debugging
        masks = masks[::-1]
        prompts = prompts[::-1]
        fg_negative_prompts = fg_negative_prompts[::-1]

    return masks, prompts, fg_negative_prompts



bg_negative = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate, headshot, close-up, partial, large, large, huge, gigantic"
fg_negative_prompt = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate, headshot, close-up, partial, large, large, huge, gigantic, cut-out, partial, occluded, weird"

sd_kw = dict(H=512, W=512)

# Max number of objects without splitting into multiple forward passes:
batch_size = 8

device = torch.device("cuda")
print(f"Using SD: {models.sd_key}")
key = models.sd_key
sd = MultiDiffusion(device, sd_version=None, batch_size=batch_size, hf_key=key)


def run(
    gen_boxes,
    bg_prompt,
    original_ind_base=None,
    bootstrapping=20,
    generate_kw=None,
    first_top=False,
    steps=50,
    guidance_scale=10.0,
    extra_neg_prompt="",
):
    print(f"gen_boxes = {gen_boxes}")
    print(f'bg_prompt = "{bg_prompt}"')
    print(f'extra_neg_prompt = "{extra_neg_prompt}"')

    gen_boxes = [
        {"name": box[0], "bounding_box": box[1]} if not isinstance(box, dict) else box
        for box in gen_boxes
    ]
    gen_boxes = filter_boxes(gen_boxes)
    show_boxes(gen_boxes, bg_prompt=bg_prompt)

    if extra_neg_prompt:
        full_bg_negative = extra_neg_prompt + ", " + bg_negative
        full_fg_negative_prompt = extra_neg_prompt + ", " + fg_negative_prompt
    else:
        full_bg_negative = bg_negative
        full_fg_negative_prompt = fg_negative_prompt

    original_masks, fg_prompts, fg_negative_prompts = boxes_to_masks_prompts(
        gen_boxes, full_fg_negative_prompt, first_top=first_top
    )

    show_masks(original_masks)
    # print(fg_prompts)

    opt = argparse.Namespace(
        seed=original_ind_base,
        steps=steps,
        bootstrapping=bootstrapping,
        **sd_kw,
        mask_paths=original_masks,
        fg_prompts=fg_prompts,
        fg_negative=fg_negative_prompts,
        bg_prompt=bg_prompt,
        bg_negative=full_bg_negative,
        guidance_scale=guidance_scale,
    )

    bg_weight = 0.0

    if opt.mask_paths:
        fg_masks = torch.cat(
            [
                preprocess_mask(mask_path, opt.H // 8, opt.W // 8, device)
                for mask_path in opt.mask_paths
            ]
        )
    else:
        fg_masks = torch.zeros(
            (0, 1, opt.H // 8, opt.W // 8), dtype=torch.float32, device=device
        )

    # bg has very low weight:
    if bg_weight == 0.0:
        print("no bg weight in box")
        bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    else:
        print(f"bg weight in box is {bg_weight}")
        bg_mask = torch.ones_like(fg_masks[0:1]) * bg_weight
    bg_mask[bg_mask < 0] = 0
    masks = torch.cat([bg_mask, fg_masks])

    # print(masks)

    prompts = [opt.bg_prompt] + opt.fg_prompts
    neg_prompts = [opt.bg_negative] + opt.fg_negative

    if generate_kw is None:
        generate_kw = {}

    gc.collect()
    torch.cuda.empty_cache()

    if opt.seed is not None:
        seed_everything(opt.seed)

    img = sd.generate(
        masks,
        prompts,
        neg_prompts,
        opt.H,
        opt.W,
        opt.steps,
        bootstrapping=opt.bootstrapping,
        guidance_scale=opt.guidance_scale,
        indep_uncond=True,
        normalization=False,
        seed=opt.seed,
        **generate_kw,
    )

    return EasyDict(image=img)


if __name__ == "__main__":
    run()
