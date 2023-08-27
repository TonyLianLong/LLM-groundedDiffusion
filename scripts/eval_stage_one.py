# This script allows evaluating stage one and saving the generated prompts to cache

import sys
import os
path_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(path_dir, '..'))

import json
import argparse
from prompt import get_prompts, prompt_types, template_versions
from utils.llm import get_llm_kwargs, get_parsed_layout, model_names
from utils.eval import get_eval_info_from_prompt, evaluate_with_boxes
from utils import cache
from tqdm import tqdm

def eval_prompt(p, prompt_type, gen_boxes, verbose=False):
    # NOTE: we use the boxes from LLM
    texts, eval_info = get_eval_info_from_prompt(p, prompt_type)
    eval_type = eval_info["type"]
    eval_success = evaluate_with_boxes(gen_boxes, eval_info, verbose=verbose)

    return eval_type, eval_success


eval_success_counts = {}
eval_all_counts = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-type", choices=prompt_types, default="demo")
    parser.add_argument("--model", choices=model_names, required=True)
    parser.add_argument("--template_version",
                        choices=template_versions, required=True)
    parser.add_argument("--skip_first_prompts", default=0, type=int)
    parser.add_argument("--num_prompts", default=None, type=int)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    template_version = args.template_version

    model, llm_kwargs = get_llm_kwargs(
        model=args.model, template_version=template_version)

    cache.cache_format = "json"
    cache.cache_path = f'{os.path.dirname(path_dir)}/cache/cache_{args.prompt_type}_{template_version}_{model}.json'
    cache.init_cache()

    prompts = get_prompts(args.prompt_type, model=model)
    print(f"Number of prompts: {len(prompts)}")

    for ind, prompt in enumerate(tqdm(prompts)):
        if isinstance(prompt, list):
            # prompt and kwargs
            prompt = prompt[0]
        prompt = prompt.strip().rstrip(".")
        if ind < args.skip_first_prompts:
            continue
        if args.num_prompts is not None and ind >= (args.skip_first_prompts + args.num_prompts):
            continue

        gen_boxes, bg_prompt, neg_prompt = get_parsed_layout(
            prompt, llm_kwargs, verbose=args.verbose)
        eval_type, eval_success = eval_prompt(
            prompt, args.prompt_type, gen_boxes, verbose=args.verbose)

        print(f"Eval success (eval_type):", eval_success)

        if eval_type not in eval_all_counts:
            eval_success_counts[eval_type] = 0
            eval_all_counts[eval_type] = 0
        eval_success_counts[eval_type] += int(eval_success)
        eval_all_counts[eval_type] += 1

    eval_success_conut, eval_all_count = 0, 0
    for k, v in eval_all_counts.items():
        print(
            f"Eval type: {k}, success: {eval_success_counts[k]}/{eval_all_counts[k]}, rate: {eval_success_counts[k]/eval_all_counts[k]:.2f}")
        eval_success_conut += eval_success_counts[k]
        eval_all_count += eval_all_counts[k]

    print(
        f"Overall: success: {eval_success_conut}/{eval_all_count}, rate: {eval_success_conut/eval_all_count:.2f}")

    if False:
        # Print what are accessed in the cache (may have multiple values in each key)
        # Not including the newly added items
        print(json.dumps(cache.cache_queries))
        print("Number of accessed keys:", len(cache.cache_queries))
