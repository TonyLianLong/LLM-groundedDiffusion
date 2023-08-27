import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
from utils.llm import get_full_model_name, model_names
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from glob import glob
from utils.eval import eval_prompt
from tqdm import tqdm
from prompt import get_prompts, prompt_types

torch.set_grad_enabled(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-type", choices=prompt_types, default="lmd")
    parser.add_argument("--model", choices=model_names, required=True)
    parser.add_argument("--run_base_path", type=str)
    parser.add_argument("--run_start_ind", default=0, type=int)
    parser.add_argument("--prompt_start_ind", default=0, type=int)
    parser.add_argument("--num_prompts", default=None, type=int)
    parser.add_argument("--skip_first_prompts", default=0, type=int)
    parser.add_argument("--detection_score_threshold",
                        default=0.05, type=float)
    parser.add_argument("--nms_threshold", default=0.5, type=float)
    parser.add_argument("--class-aware-nms", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--no-cuda", action='store_true')
    args = parser.parse_args()

    model = get_full_model_name(args.model)
    prompts = get_prompts(args.prompt_type, model=model)

    print(f"Number of prompts: {len(prompts)}")

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_vit_model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32")
    owl_vit_model.eval()

    use_cuda = not args.no_cuda

    if use_cuda:
        owl_vit_model.cuda()

    eval_success_counts = {}
    eval_all_counts = {}

    for ind, prompt in enumerate(tqdm(prompts)):
        if isinstance(prompt, list):
            # prompt and kwargs
            prompt = prompt[0]
        if ind < args.skip_first_prompts:
            continue
        if args.num_prompts is not None and ind >= (args.skip_first_prompts + args.num_prompts):
            continue

        search_path = f"{args.run_base_path}/{ind+args.run_start_ind}/img_*.png"

        # NOTE: sorted with string type
        path = sorted(glob(search_path))
        if len(path) == 0:
            print(f"***No image matching {search_path}, skipping***")
            continue
        elif len(path) > 1:
            print(
                f"***More than one images match {search_path}: {path}, skipping***")
            continue
        path = path[0]
        print(f"Image path: {path}")

        eval_type, eval_success = eval_prompt(prompt, args.prompt_type, path, processor, owl_vit_model, score_threshold=args.detection_score_threshold,
                                              nms_threshold=args.nms_threshold, use_class_aware_nms=args.class_aware_nms, use_cuda=use_cuda, verbose=args.verbose)

        print(f"Eval success (eval_type):", eval_success)

        if eval_type not in eval_all_counts:
            eval_success_counts[eval_type] = 0
            eval_all_counts[eval_type] = 0
        eval_success_counts[eval_type] += int(eval_success)
        eval_all_counts[eval_type] += 1

    summary = []
    eval_success_conut, eval_all_count = 0, 0
    for k, v in eval_all_counts.items():
        rate = eval_success_counts[k]/eval_all_counts[k]
        print(
            f"Eval type: {k}, success: {eval_success_counts[k]}/{eval_all_counts[k]}, rate: {round(rate, 2):.2f}")
        eval_success_conut += eval_success_counts[k]
        eval_all_count += eval_all_counts[k]
        summary.append(rate)

    rate = eval_success_conut/eval_all_count
    print(
        f"Overall: success: {eval_success_conut}/{eval_all_count}, rate: {rate:.2f}")
    summary.append(rate)

    summary_str = '/'.join([f"{round(rate, 2):.2f}" for rate in summary])
    print(f"Summary: {summary_str}")
