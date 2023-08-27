import requests
from prompt import templates, stop
from easydict import EasyDict
from utils.cache import get_cache, add_cache
from utils.parse import size, parse_input_with_negative, filter_boxes
import traceback
import time

model_names = ["vicuna", "vicuna-13b", "vicuna-13b-v1.3", "vicuna-33b-v1.3", "Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf", "FreeWilly2", "StableBeluga2", "gpt-3.5-turbo", "gpt-3.5", "gpt-4", "text-davinci-003"]

def get_full_prompt(template, prompt, suffix=None):
    full_prompt = template.format(prompt=prompt)
    if suffix:
        full_prompt += suffix
    return full_prompt

def get_full_model_name(model):
    if model == "gpt-3.5":
        model = "gpt-3.5-turbo"
    elif model == "vicuna":
        model = "vicuna-13b"
    elif model == "gpt-4":
        model = "gpt-4"
        
    return model

def get_llm_kwargs(model, template_version):
    model = get_full_model_name(model)
        
    print(f"Using template: {template_version}")

    template = templates[template_version]

    if "vicuna" in model.lower() or "llama" in model.lower() or "freewilly" in model.lower() or "stablebeluga2" in model.lower():
        api_base = "http://localhost:8000/v1"
        max_tokens = 900
        temperature = 0.25
        headers = {}
    else:
        from utils.api_key import api_key
        
        api_base = "https://api.openai.com/v1"
        max_tokens = 900
        temperature = 0.25
        headers = {"Authorization": f"Bearer {api_key}"}

    llm_kwargs = EasyDict(model=model, template=template, api_base=api_base, max_tokens=max_tokens, temperature=temperature, headers=headers, stop=stop)

    return model, llm_kwargs


def get_layout(prompt, llm_kwargs, suffix=""):
    # No cache in this function
    model, template, api_base, max_tokens, temperature, stop, headers = llm_kwargs.model, llm_kwargs.template, llm_kwargs.api_base, llm_kwargs.max_tokens, llm_kwargs.temperature, llm_kwargs.stop, llm_kwargs.headers

    done = False
    attempts = 0
    while not done:
        if "gpt" in model:
            r = requests.post(f'{api_base}/chat/completions', json={
                "model": model,
                "messages": [{"role": "user", "content": get_full_prompt(template, prompt, suffix).strip()}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }, headers=headers)
        else:
            r = requests.post(f'{api_base}/completions', json={
                "model": model,
                "prompt": get_full_prompt(template, prompt, suffix).strip(),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }, headers=headers)

        done = r.status_code == 200

        if not done:
            print(r.json())
            attempts += 1
        if attempts >= 3 and "gpt" in model:
            print("Retrying after 1 minute")
            time.sleep(60)
        if attempts >= 5 and "gpt" in model:
            print("Exiting due to many non-successful attempts")
            exit()

    if "gpt" in model:
        response = r.json()['choices'][0]['message']['content']
    else:
        response = r.json()['choices'][0]['text']

    return response


def get_layout_with_cache(prompt, *args, **kwargs):
    # Note that cache path needs to be set correctly, as get_cache does not check whether the cache is generated with the given model in the given setting.

    response = get_cache(prompt)

    if response is not None:
        print(f"Cache hit: {prompt}")
        return response

    print(f"Cache miss: {prompt}")
    response = get_layout(prompt, *args, **kwargs)

    add_cache(prompt, response)

    return response


def get_parsed_layout(prompt, llm_kwargs, verbose=True):
    done = False

    while not done:
        try:
            layout_text = get_layout_with_cache(prompt, llm_kwargs)

            if verbose:
                print(layout_text)
            gen_boxes, bg_prompt, neg_prompt = parse_input_with_negative(layout_text, no_input=True)

            if len(gen_boxes) > 0:
                gen_boxes = [{'name': box[0], 'bounding_box': box[1]}
                             for box in gen_boxes]
                gen_boxes = filter_boxes(gen_boxes, scale_boxes=False)
        except Exception as e:
            print(f"Error: {e}, retrying")
            traceback.print_exc()
            continue

        done = True

    if verbose:
        print(f"gen_boxes = {gen_boxes}")
        print(f"bg_prompt = \"{bg_prompt}\"")
        print(f"neg_prompt = \"{neg_prompt}\"")

    return gen_boxes, bg_prompt, neg_prompt
