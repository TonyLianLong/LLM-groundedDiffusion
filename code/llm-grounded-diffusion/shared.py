from models import load_sd, sam


DEFAULT_SO_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate, two, many, group, occlusion, occluded, side, border, collate"
DEFAULT_OVERALL_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"


use_fp16 = False

sd_key = "gligen/diffusers-generation-text-box"

print(f"Using SD: {sd_key}")
model_dict = load_sd(key=sd_key, use_fp16=use_fp16, load_inverse_scheduler=False)

sam_model_dict = sam.load_sam()
