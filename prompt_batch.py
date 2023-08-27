import os
from prompt import get_prompts, prompt_types, template_versions
from utils import parse
from utils.parse import parse_input_with_negative, bg_prompt_text, neg_prompt_text, filter_boxes, show_boxes
from utils.llm import get_llm_kwargs, get_full_prompt, get_layout, model_names
from utils import cache
import matplotlib.pyplot as plt
import argparse
import time

# This only applies to visualization in this file.
scale_boxes = False

if scale_boxes:
    print("Scaling the bounding box to fit the scene")
else:
    print("Not scaling the bounding box to fit the scene")

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--prompt-type", choices=prompt_types, default="demo")
    parser.add_argument("--model", choices=model_names, required=True)
    parser.add_argument("--template_version", choices=template_versions, required=True)
    parser.add_argument("--auto-query", action='store_true', help='Auto query using the API')
    parser.add_argument("--always-save", action='store_true', help='Always save the layout without confirming')
    parser.add_argument("--no-visualize", action='store_true', help='No visualizations')
    parser.add_argument("--visualize-cache-hit", action='store_true', help='Save boxes for cache hit')
    args = parser.parse_args()
    
    visualize_cache_hit = args.visualize_cache_hit
    
    template_version = args.template_version
    
    model, llm_kwargs = get_llm_kwargs(
        model=args.model, template_version=template_version)
    template = llm_kwargs.template

    # This is for visualizing bounding boxes
    parse.img_dir = f"img_generations/imgs_{args.prompt_type}_template{template_version}"
    if not args.no_visualize:
        os.makedirs(parse.img_dir, exist_ok=True)

    cache.cache_path = f'cache/cache_{args.prompt_type.replace("lmd_", "")}{"_" + template_version if args.template_version != "v5" else ""}_{model}.json'
    print(f"Cache: {cache.cache_path}")
    os.makedirs(os.path.dirname(cache.cache_path), exist_ok=True)
    cache.cache_format = "json"

    cache.init_cache()

    prompts_query = get_prompts(args.prompt_type, model=model)
    
    for ind, prompt in enumerate(prompts_query):
        if isinstance(prompt, list):
            # prompt, seed
            prompt = prompt[0]
        prompt = prompt.strip().rstrip(".")
        
        response = cache.get_cache(prompt)
        if response is None:
            print(f"Cache miss: {prompt}")
            
            if not args.auto_query:
                print("#########")
                prompt_full = get_full_prompt(template=template, prompt=prompt)
                print(prompt_full, end="")
                print("#########")
                resp = None
            
            attempts = 0
            while True:
                attempts += 1
                if args.auto_query:
                    resp = get_layout(prompt=prompt, llm_kwargs=llm_kwargs)
                    print("Response:", resp)
                
                try:
                    parsed_input = parse_input_with_negative(text=resp, no_input=args.auto_query)
                    if parsed_input is None:
                        raise ValueError("Invalid input")
                    raw_gen_boxes, bg_prompt, neg_prompt = parsed_input
                except (ValueError, SyntaxError, TypeError) as e:
                    if attempts > 3:
                        print("Retrying too many times, skipping")
                        break
                    print(f"Encountered invalid data with prompt {prompt} and response {resp}: {e}, retrying")
                    time.sleep(10)
                    continue
                
                gen_boxes = [{'name': box[0], 'bounding_box': box[1]} for box in raw_gen_boxes]
                gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes)
                if not args.no_visualize:
                    show_boxes(gen_boxes, bg_prompt=bg_prompt, neg_prompt=neg_prompt, ind=ind)
                    plt.clf()
                    print(f"Visualize masks at {parse.img_dir}")
                if not args.always_save:
                    save = input("Save (y/n)? ").strip()
                else:
                    save = "y"
                if save == "y" or save == "Y":
                    response = f"{raw_gen_boxes}\n{bg_prompt_text}{bg_prompt}\n{neg_prompt_text}{neg_prompt}"
                    cache.add_cache(prompt, response)
                else:
                    print("Not saved. Will generate the same prompt again.")
                    continue
                break
        else:
            print(f"Cache hit: {prompt}")
            
            if visualize_cache_hit:
                raw_gen_boxes, bg_prompt, neg_prompt = parse_input_with_negative(text=response)
                
                gen_boxes = [{'name': box[0], 'bounding_box': box[1]} for box in raw_gen_boxes]
                gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes)
                show_boxes(gen_boxes, bg_prompt=bg_prompt, neg_prompt=neg_prompt, ind=ind)
                plt.clf()
                print(f"Visualize masks at {parse.img_dir}")

