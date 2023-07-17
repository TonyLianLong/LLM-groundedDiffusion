import gradio as gr
import numpy as np
import ast
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from utils.parse import filter_boxes
from generation import run as run_ours
from baseline import run as run_baseline
import torch
from shared import DEFAULT_SO_NEGATIVE_PROMPT, DEFAULT_OVERALL_NEGATIVE_PROMPT
from examples import stage1_examples, stage2_examples

print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

box_scale = (512, 512)
size = box_scale

bg_prompt_text = "Background prompt: "

default_template = """You are an intelligent bounding box generator. I will provide you with a caption for a photo, image, or painting. Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene. The images are of size 512x512, and the bounding boxes should not overlap or go beyond the image boundaries. Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and include exactly one object. Make the boxes larger if possible. Do not put objects that are already provided in the bounding boxes into the background prompt. If needed, you can make reasonable guesses. Generate the object descriptions and background prompts in English even if the caption might not be in English. Do not include non-existing or excluded objects in the background prompt. Please refer to the example below for the desired format.

Caption: A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky
Objects: [('a green car', [21, 181, 211, 159]), ('a blue truck', [269, 181, 209, 160]), ('a red air balloon', [66, 8, 145, 135]), ('a bird', [296, 42, 143, 100])]
Background prompt: A realistic image of a landscape scene

Caption: A watercolor painting of a wooden table in the living room with an apple on it
Objects: [('a wooden table', [65, 243, 344, 206]), ('a apple', [206, 306, 81, 69])]
Background prompt: A watercolor painting of a living room

Caption: A watercolor painting of two pandas eating bamboo in a forest
Objects: [('a panda eating bambooo', [30, 171, 212, 226]), ('a panda eating bambooo', [264, 173, 222, 221])]
Background prompt: A watercolor painting of a forest

Caption: A realistic image of four skiers standing in a line on the snow near a palm tree
Objects: [('a skier', [5, 152, 139, 168]), ('a skier', [278, 192, 121, 158]), ('a skier', [148, 173, 124, 155]), ('a palm tree', [404, 180, 103, 180])]
Background prompt: A realistic image of an outdoor scene with snow

Caption: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
Objects: [('a steam boat', [232, 225, 257, 149]), ('a jumping pink dolphin', [21, 249, 189, 123])]
Background prompt: An oil painting of the sea

Caption: A realistic image of a cat playing with a dog in a park with flowers
Objects: [('a playful cat', [51, 67, 271, 324]), ('a playful dog', [302, 119, 211, 228])]
Background prompt: A realistic image of a park with flowers

Caption: 一个客厅场景的油画，墙上挂着电视，电视下面是一个柜子，柜子上有一个花瓶。
Objects: [('a tv', [88, 85, 335, 203]), ('a cabinet', [57, 308, 404, 201]), ('a flower vase', [166, 222, 92, 108])]
Background prompt: An oil painting of a living room scene"""

simplified_prompt = """{template}

Caption: {prompt}
Objects: """

prompt_placeholder = "A realistic photo of a gray cat and an orange dog on the grass."

layout_placeholder = """Caption: A realistic photo of a gray cat and an orange dog on the grass.
Objects: [('a gray cat', [67, 243, 120, 126]), ('an orange dog', [265, 193, 190, 210])]
Background prompt: A realistic photo of a grassy area."""

def get_lmd_prompt(prompt, template=default_template):
    if prompt == "":
        prompt = prompt_placeholder
    if template == "":
        template = default_template
    return simplified_prompt.format(template=template, prompt=prompt)

def get_layout_image(response):
    if response == "":
        response = layout_placeholder
    gen_boxes, bg_prompt = parse_input(response)
    fig = plt.figure(figsize=(8, 8))
    # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
    show_boxes(gen_boxes, bg_prompt)
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return data

def get_layout_image_gallery(response):
    return [get_layout_image(response)]

def get_ours_image(response, overall_prompt_override="", seed=0, num_inference_steps=20, dpm_scheduler=True, use_autocast=False, fg_seed_start=20, fg_blending_ratio=0.1, frozen_step_ratio=0.4, gligen_scheduled_sampling_beta=0.3, so_negative_prompt=DEFAULT_SO_NEGATIVE_PROMPT, overall_negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT, show_so_imgs=False, scale_boxes=False):
    if response == "":
        response = layout_placeholder
    gen_boxes, bg_prompt = parse_input(response)
    gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes)
    spec = {
        # prompt is unused
        'prompt': '',
        'gen_boxes': gen_boxes,
        'bg_prompt': bg_prompt
    }
    
    if dpm_scheduler:
        scheduler_key = "dpm_scheduler"
    else:
        scheduler_key = "scheduler"
        
    image_np, so_img_list = run_ours(
        spec, bg_seed=seed, overall_prompt_override=overall_prompt_override, fg_seed_start=fg_seed_start, 
        fg_blending_ratio=fg_blending_ratio,frozen_step_ratio=frozen_step_ratio, use_autocast=use_autocast,
        gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta, num_inference_steps=num_inference_steps, scheduler_key=scheduler_key,
        so_negative_prompt=so_negative_prompt, overall_negative_prompt=overall_negative_prompt, so_batch_size=2
    )
    images = [image_np]
    if show_so_imgs:
        images.extend([np.asarray(so_img) for so_img in so_img_list])
    return images

def get_baseline_image(prompt, seed=0):
    if prompt == "":
        prompt = prompt_placeholder
    
    scheduler_key = "dpm_scheduler"
    num_inference_steps = 20
    
    image_np = run_baseline(prompt, bg_seed=seed, scheduler_key=scheduler_key, num_inference_steps=num_inference_steps)
    return [image_np]

def parse_input(text=None):
    try:
        if "Objects: " in text:
            text = text.split("Objects: ")[1]
            
        text_split = text.split(bg_prompt_text)
        if len(text_split) == 2:
            gen_boxes, bg_prompt = text_split
        gen_boxes = ast.literal_eval(gen_boxes)    
        bg_prompt = bg_prompt.strip()
    except Exception as e:
        raise gr.Error(f"response format invalid: {e} (text: {text})")
    
    return gen_boxes, bg_prompt

def draw_boxes(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4)
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)

        # print(ann)
        name = ann['name'] if 'name' in ann else str(ann['category_id'])
        ax.text(bbox_x, bbox_y, name, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

    p = PatchCollection(polygons, facecolor='none',
                        edgecolors=color, linewidths=2)
    ax.add_collection(p)


def show_boxes(gen_boxes, bg_prompt=None):
    anns = [{'name': gen_box[0], 'bbox': gen_box[1]}
            for gen_box in gen_boxes]

    # White background (to allow line to show on the edge)
    I = np.ones((size[0]+4, size[1]+4, 3), dtype=np.uint8) * 255

    plt.imshow(I)
    plt.axis('off')

    if bg_prompt is not None:
        ax = plt.gca()
        ax.text(0, 0, bg_prompt, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

        c = np.zeros((1, 3))
        [bbox_x, bbox_y, bbox_w, bbox_h] = (0, 0, size[1], size[0])
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons = [Polygon(np_poly)]
        color = [c]
        p = PatchCollection(polygons, facecolor='none',
                            edgecolors=color, linewidths=2)
        ax.add_collection(p)

    draw_boxes(anns)

duplicate_html = '<a style="display:inline-block" href="https://huggingface.co/spaces/longlian/llm-grounded-diffusion?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>'

html = f"""<h1>LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models</h1>
            <h2>LLM + Stable Diffusion => better prompt understanding in text2image generation 🤩</h2>
            <h2><a href='https://llm-grounded-diffusion.github.io/'>Project Page</a> | <a href='https://bair.berkeley.edu/blog/2023/05/23/lmd/'>5-minute Blog Post</a> | <a href='https://arxiv.org/pdf/2305.13655.pdf'>ArXiv Paper</a> | <a href='https://github.com/TonyLianLong/LLM-groundedDiffusion'>Github</a> | <a href='https://llm-grounded-diffusion.github.io/#citation'>Cite our work</a> if our ideas inspire you.</h2>
            <p><b>Tips:</b><p>
            <p>1. If ChatGPT doesn't generate layout, add/remove the trailing space (added by default) and/or use GPT-4.</p>
            <p>2. You can perform multi-round specification by giving ChatGPT follow-up requests (e.g., make the object boxes bigger).</p>
            <p>3. You can also try prompts in Simplified Chinese. If you want to try prompts in another language, translate the first line of last example to your language.</p>
            <p>4. The diffusion model only runs 20 steps by default in this demo. You can make it run more steps to get higher quality images (or tweak frozen steps/guidance steps for better guidance and coherence).</p>
            <br/>
            <p>Implementation note: In this demo, we replace the attention manipulation in our layout-guided Stable Diffusion described in our paper with GLIGEN due to much faster inference speed (<b>FlashAttention supported, no backprop needed</b> during inference). Compared to vanilla GLIGEN, we have better coherence. Other parts of text-to-image pipeline, including single object generation and SAM, remain the same. The settings and examples in the prompt are simplified in this demo.</p>
            <style>.btn {{flex-grow: unset !important;}} </style>
            """

with gr.Blocks(
    title="LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models"
) as g:
    gr.HTML(html)
    with gr.Tab("Stage 1. Image Prompt to ChatGPT"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(lines=2, label="Prompt for Layout Generation", placeholder=prompt_placeholder)
                generate_btn = gr.Button("Generate Prompt", variant='primary', elem_classes="btn")
                with gr.Accordion("Advanced options", open=False):
                    template = gr.Textbox(lines=10, label="Custom Template", placeholder="Customized Template", value=default_template)
            with gr.Column(scale=1):
                output = gr.Textbox(label="Paste this into ChatGPT (GPT-4 preferred; on Mac, click text and press Command+A and Command+C to copy all)", show_copy_button=True)
                gr.HTML("<a href='https://chat.openai.com' target='_blank'>Click here to open ChatGPT</a>")
        generate_btn.click(fn=get_lmd_prompt, inputs=[prompt, template], outputs=output, api_name="get_lmd_prompt")
    
        gr.Examples(
            examples=stage1_examples,
            inputs=[prompt],
            outputs=[output],
            fn=get_lmd_prompt,
            cache_examples=True
        )
    
    with gr.Tab("Stage 2 (New). Layout to Image generation"):
        with gr.Row():
            with gr.Column(scale=1):
                response = gr.Textbox(lines=8, label="Paste ChatGPT response here (no original caption needed)", placeholder=layout_placeholder)
                overall_prompt_override = gr.Textbox(lines=2, label="Prompt for overall generation (optional but recommended)", placeholder="You can put your input prompt for layout generation here, helpful if your scene cannot be represented by background prompt and boxes only, e.g., with object interactions. If left empty: background prompt with [objects].", value="")
                num_inference_steps = gr.Slider(1, 250, value=20, step=1, label="Number of denoising steps (set to >=50 for higher generation quality)")
                seed = gr.Slider(0, 10000, value=0, step=1, label="Seed")
                with gr.Accordion("Advanced options (play around for better generation)", open=False):
                    frozen_step_ratio = gr.Slider(0, 1, value=0.4, step=0.1, label="Foreground frozen steps ratio (higher: preserve object attributes; lower: higher coherence; set to 0: (almost) equivalent to vanilla GLIGEN except details)")
                    gligen_scheduled_sampling_beta = gr.Slider(0, 1, value=0.3, step=0.1, label="GLIGEN guidance steps ratio (the beta value)")
                    dpm_scheduler = gr.Checkbox(label="Use DPM scheduler (unchecked: DDIM scheduler, may have better coherence, recommend >=50 inference steps)", show_label=False, value=True)
                    use_autocast = gr.Checkbox(label="Use FP16 Mixed Precision (faster but with slightly lower quality)", show_label=False, value=True)
                    fg_seed_start = gr.Slider(0, 10000, value=20, step=1, label="Seed for foreground variation")
                    fg_blending_ratio = gr.Slider(0, 1, value=0.1, step=0.01, label="Variations added to foreground for single object generation (0: no variation, 1: max variation)")
                    so_negative_prompt = gr.Textbox(lines=1, label="Negative prompt for single object generation", value=DEFAULT_SO_NEGATIVE_PROMPT)
                    overall_negative_prompt = gr.Textbox(lines=1, label="Negative prompt for overall generation", value=DEFAULT_OVERALL_NEGATIVE_PROMPT)
                    show_so_imgs = gr.Checkbox(label="Show annotated single object generations", show_label=False, value=False)
                    scale_boxes = gr.Checkbox(label="Scale bounding boxes to just fit the scene", show_label=False, value=False)
                visualize_btn = gr.Button("Visualize Layout", elem_classes="btn")
                generate_btn = gr.Button("Generate Image from Layout", variant='primary', elem_classes="btn")
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Generated image", show_label=False, elem_id="gallery", columns=[1], rows=[1], object_fit="contain", preview=True
                )
        visualize_btn.click(fn=get_layout_image_gallery, inputs=response, outputs=gallery, api_name="visualize-layout")
        generate_btn.click(fn=get_ours_image, inputs=[response, overall_prompt_override, seed, num_inference_steps, dpm_scheduler, use_autocast, fg_seed_start, fg_blending_ratio, frozen_step_ratio, gligen_scheduled_sampling_beta, so_negative_prompt, overall_negative_prompt, show_so_imgs, scale_boxes], outputs=gallery, api_name="layout-to-image")

        gr.Examples(
            examples=stage2_examples,
            inputs=[response, overall_prompt_override, seed],
            outputs=[gallery],
            fn=get_ours_image,
            cache_examples=True
        )

    with gr.Tab("Baseline: Stable Diffusion"):
        with gr.Row():
            with gr.Column(scale=1):
                sd_prompt = gr.Textbox(lines=2, label="Prompt for baseline SD", placeholder=prompt_placeholder)
                seed = gr.Slider(0, 10000, value=0, step=1, label="Seed")
                generate_btn = gr.Button("Generate", elem_classes="btn")
            # with gr.Column(scale=1):
            #     output = gr.Image(shape=(512, 512), elem_classes="img", elem_id="img")
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Generated image", show_label=False, elem_id="gallery2", columns=[1], rows=[1], object_fit="contain", preview=True
                )
        generate_btn.click(fn=get_baseline_image, inputs=[sd_prompt, seed], outputs=gallery, api_name="baseline")

        gr.Examples(
            examples=stage1_examples,
            inputs=[sd_prompt],
            outputs=[gallery],
            fn=get_baseline_image,
            cache_examples=True
        )

g.launch()
