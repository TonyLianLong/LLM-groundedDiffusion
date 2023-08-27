import ast
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import warnings
import inflect
import gradio as gr

p = inflect.engine()
# user_error = ValueError
user_error = gr.Error

img_dir = "imgs"
objects_text = "Objects: "
bg_prompt_text = "Background prompt: "
bg_prompt_text_no_trailing_space = bg_prompt_text.rstrip()
neg_prompt_text = "Negative prompt: "
neg_prompt_text_no_trailing_space = neg_prompt_text.rstrip()

# h, w
box_scale = (512, 512)
size = box_scale
size_h, size_w = size
print(f"Using box scale: {box_scale}")


def parse_input(text=None, no_input=False):
    warnings.warn("Parsing input without negative prompt is deprecated.")
    
    if not text:
        if no_input:
            raise user_error(f"No input parsed in \"{text}\".")
        
        text = input("Enter the response: ")
    if objects_text in text:
        text = text.split(objects_text)[1]
        
    text_split = text.split(bg_prompt_text_no_trailing_space)
    if len(text_split) == 2:
        gen_boxes, bg_prompt = text_split
    elif len(text_split) == 1:
        if no_input:
            raise user_error(f"Invalid input (no background prompt): {text}")
        gen_boxes = text
        bg_prompt = ""
        while not bg_prompt:
            # Ignore the empty lines in the response
            bg_prompt = input("Enter the background prompt: ").strip()
        if bg_prompt_text_no_trailing_space in bg_prompt:
            bg_prompt = bg_prompt.split(bg_prompt_text_no_trailing_space)[1]
    else:
        raise user_error(f"Invalid input (possibly multiple background prompts): {text}")
    try:
        gen_boxes = ast.literal_eval(gen_boxes)    
    except SyntaxError as e:
        # Sometimes the response is in plain text
        if "No objects" in gen_boxes:
            gen_boxes = []
        else:
            raise e
    bg_prompt = bg_prompt.strip()
    
    return gen_boxes, bg_prompt

def parse_input_with_negative(text=None, no_input=False):
    # no_input: should not request interactive input
    
    if not text:
        if no_input:
            raise user_error(f"No input parsed in \"{text}\".")
        
        text = input("Enter the response: ")
    if objects_text in text:
        text = text.split(objects_text)[1]
        
    text_split = text.split(bg_prompt_text_no_trailing_space)
    if len(text_split) == 2:
        gen_boxes, text_rem = text_split
    elif len(text_split) == 1:
        if no_input:
            raise user_error(f"Invalid input (no background prompt): {text}")
        gen_boxes = text
        text_rem = ""
        while not text_rem:
            # Ignore the empty lines in the response
            text_rem = input("Enter the background prompt: ").strip()
        if bg_prompt_text_no_trailing_space in text_rem:
            text_rem = text_rem.split(bg_prompt_text_no_trailing_space)[1]
    else:
        raise user_error(f"Invalid input (possibly multiple background prompts): {text}")
    
    text_split = text_rem.split(neg_prompt_text_no_trailing_space)
    
    if len(text_split) == 2:
        bg_prompt, neg_prompt = text_split
    elif len(text_split) == 1:
        bg_prompt = text_rem
        # Negative prompt is optional: if it's not provided, we default to empty string
        neg_prompt = ""
        if not no_input:
            # Ignore the empty lines in the response
            neg_prompt = input("Enter the negative prompt: ").strip()
            if neg_prompt_text_no_trailing_space in neg_prompt:
                neg_prompt = neg_prompt.split(neg_prompt_text_no_trailing_space)[1]
    else:
        raise user_error(f"Invalid input (possibly multiple negative prompts): {text}")
    
    try:
        gen_boxes = ast.literal_eval(gen_boxes)    
    except SyntaxError as e:
        # Sometimes the response is in plain text
        if "No objects" in gen_boxes or gen_boxes.strip() == "":
            gen_boxes = []
        else:
            raise e
    bg_prompt = bg_prompt.strip()
    neg_prompt = neg_prompt.strip()
    
    # LLM may return "None" to mean no negative prompt provided.
    if neg_prompt == "None":
        neg_prompt = ""
    
    return gen_boxes, bg_prompt, neg_prompt

def filter_boxes(gen_boxes, scale_boxes=True, ignore_background=True, max_scale=3):
    if gen_boxes is None:
        return []
    
    if len(gen_boxes) == 0:
        return []
    
    box_dict_format = False
    gen_boxes_new = []
    for gen_box in gen_boxes:
        if isinstance(gen_box, dict):
            if not gen_box['bounding_box']:
                continue
            name, [bbox_x, bbox_y, bbox_w, bbox_h] = gen_box['name'], gen_box['bounding_box']
            box_dict_format = True
        else:
            if not gen_box[1]:
                continue
            name, [bbox_x, bbox_y, bbox_w, bbox_h] = gen_box
        if bbox_w <= 0 or bbox_h <= 0:
            # Empty boxes
            continue
        if ignore_background:
            if (bbox_w >= size[1] and bbox_h >= size[0]) or bbox_x > size[1] or bbox_y > size[0]:
                # Ignore the background boxes
                continue
        
        if bbox_x < 0 or bbox_y < 0 or bbox_x + bbox_w > size[1] or bbox_y + bbox_h > size[0]:
            # Out of bounds boxes exist: we need to scale and shift all the boxes
            print(f"**Some boxes are out of bounds: {gen_box}, scaling all the boxes to fit**")
            scale_boxes = True

        gen_boxes_new.append(gen_box)
    
    gen_boxes = gen_boxes_new
    
    if len(gen_boxes) == 0:
        return []
    
    filtered_gen_boxes = []
    if box_dict_format:
        # For compatibility
        bbox_left_x_min = min([gen_box['bounding_box'][0] for gen_box in gen_boxes])
        bbox_right_x_max = max([gen_box['bounding_box'][0] + gen_box['bounding_box'][2] for gen_box in gen_boxes])
        bbox_top_y_min = min([gen_box['bounding_box'][1] for gen_box in gen_boxes])
        bbox_bottom_y_max = max([gen_box['bounding_box'][1] + gen_box['bounding_box'][3] for gen_box in gen_boxes])
    else:
        bbox_left_x_min = min([gen_box[1][0] for gen_box in gen_boxes])
        bbox_right_x_max = max([gen_box[1][0] + gen_box[1][2] for gen_box in gen_boxes])
        bbox_top_y_min = min([gen_box[1][1] for gen_box in gen_boxes])
        bbox_bottom_y_max = max([gen_box[1][1] + gen_box[1][3] for gen_box in gen_boxes])
    
    # All boxes are empty
    if (bbox_right_x_max - bbox_left_x_min) == 0:
        return []
    
    # Used if scale_boxes is True
    shift = -bbox_left_x_min
    # Make sure the boxes fit horizontally and vertically
    scale_w = size_w / (bbox_right_x_max - bbox_left_x_min)
    scale_h = size_h / (bbox_bottom_y_max - bbox_top_y_min)
    
    scale = min(scale_w, scale_h, max_scale)
    
    for gen_box in gen_boxes:
        if box_dict_format:
            name, [bbox_x, bbox_y, bbox_w, bbox_h] = gen_box['name'], gen_box['bounding_box']
        else:
            name, [bbox_x, bbox_y, bbox_w, bbox_h] = gen_box
            
        if scale_boxes:
            # Vertical: move the boxes if out of bound
            # Horizontal: move and scale the boxes so it spans the horizontal line
            
            bbox_x = (bbox_x + shift) * scale
            bbox_y = bbox_y * scale
            bbox_w, bbox_h = bbox_w * scale, bbox_h * scale
            # TODO: verify this makes the y center not moving
            bbox_y_offset = 0
            if bbox_top_y_min * scale + bbox_y_offset < 0:
                bbox_y_offset -= bbox_top_y_min * scale
            if bbox_bottom_y_max * scale + bbox_y_offset >= size_h:
                bbox_y_offset -= bbox_bottom_y_max * scale - size_h
            bbox_y += bbox_y_offset
            
            if bbox_y < 0:
                bbox_y, bbox_h = 0, bbox_h - bbox_y
                
        name = name.rstrip(".")
        bounding_box = (int(np.round(bbox_x)), int(np.round(bbox_y)), int(np.round(bbox_w)), int(np.round(bbox_h)))
        if box_dict_format:
            gen_box = {
                'name': name,
                'bounding_box': bounding_box
            }
        else:
            gen_box = (name, bounding_box)
        
        filtered_gen_boxes.append(gen_box)
        
    return filtered_gen_boxes

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


def show_boxes(gen_boxes, bg_prompt=None, neg_prompt=None, ind=None, show=False, save=False):
    if len(gen_boxes) == 0:
        return
    
    if isinstance(gen_boxes[0], dict):
        anns = [{'name': gen_box['name'], 'bbox': gen_box['bounding_box']}
                for gen_box in gen_boxes]
    else:
        anns = [{'name': gen_box[0], 'bbox': gen_box[1]} for gen_box in gen_boxes]

    # White background (to allow line to show on the edge)
    I = np.ones((size[0]+4, size[1]+4, 3), dtype=np.uint8) * 255

    plt.imshow(I)
    plt.axis('off')

    if bg_prompt is not None:
        ax = plt.gca()
        ax.text(0, 0, bg_prompt + f"(Neg: {neg_prompt})" if neg_prompt else bg_prompt, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

        c = (np.zeros((1, 3)))
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
    if show:
        plt.show()
    else:
        print("Saved boxes visualizations to", f"{img_dir}/boxes.png", f"ind: {ind}")
        if ind is not None:
            plt.savefig(f"{img_dir}/boxes_{ind}.png")
        plt.savefig(f"{img_dir}/boxes.png")

def show_masks(masks):
    masks_to_show = np.zeros((*size, 3), dtype=np.float32)
    for mask in masks:
        c = (np.random.random((3,))*0.6+0.4)

        masks_to_show += mask[..., None] * c[None, None, :]
    plt.imshow(masks_to_show)
    plt.savefig(f"{img_dir}/masks.png")
    plt.show()
    plt.clf()

def convert_box(box, height, width):
    # box: x, y, w, h (in 512 format) -> x_min, y_min, x_max, y_max
    x_min, y_min = box[0] / width, box[1] / height
    w_box, h_box = box[2] / width, box[3] / height
    
    x_max, y_max = x_min + w_box, y_min + h_box
    
    return x_min, y_min, x_max, y_max

def convert_spec(spec, height, width, include_counts=True, verbose=False):
    # Infer from spec
    prompt, gen_boxes, bg_prompt = spec['prompt'], spec['gen_boxes'], spec['bg_prompt']
    
    # This ensures the same objects appear together because flattened `overall_phrases_bboxes` should EXACTLY correspond to `so_prompt_phrase_box_list`. 
    gen_boxes = sorted(gen_boxes, key=lambda gen_box: gen_box[0])
    
    gen_boxes = [(name, convert_box(box, height=height, width=width)) for name, box in gen_boxes]
    
    # NOTE: so phrase should include all the words associated to the object (otherwise "an orange dog" may be recognized as "an orange" by the model generating the background).
    # so word should have one token that includes the word to transfer cross attention (the object name).
    # Currently using the last word of the object name as word.
    if bg_prompt:
        so_prompt_phrase_word_box_list = [(f"{bg_prompt} with {name}", name, name.split(" ")[-1], box) for name, box in gen_boxes]
    else:
        so_prompt_phrase_word_box_list = [(f"{name}", name, name.split(" ")[-1], box) for name, box in gen_boxes]
    
    objects = [gen_box[0] for gen_box in gen_boxes]
    
    objects_unique, objects_count = np.unique(objects, return_counts=True)

    num_total_matched_boxes = 0
    overall_phrases_words_bboxes = []
    for ind, object_name in enumerate(objects_unique):
        bboxes = [box for name, box in gen_boxes if name == object_name]
        
        if objects_count[ind] > 1:
            phrase = p.plural_noun(object_name.replace("an ", "").replace("a ", ""))
            if include_counts:
                phrase = p.number_to_words(objects_count[ind]) + " " + phrase
        else:
            phrase = object_name
        # Currently using the last word of the phrase as word.
        word = phrase.split(' ')[-1]
        
        num_total_matched_boxes += len(bboxes)
        overall_phrases_words_bboxes.append((phrase, word, bboxes))
        
    assert num_total_matched_boxes == len(gen_boxes), f"{num_total_matched_boxes} != {len(gen_boxes)}"

    objects_str = ", ".join([phrase for phrase, _, _ in overall_phrases_words_bboxes])
    if objects_str:
        if bg_prompt:
            overall_prompt = f"{bg_prompt} with {objects_str}"
        else:
            overall_prompt = objects_str
    else:
        overall_prompt = bg_prompt
        
    if verbose:
        print("so_prompt_phrase_word_box_list:", so_prompt_phrase_word_box_list)
        print("overall_prompt:", overall_prompt)
        print("overall_phrases_words_bboxes:", overall_phrases_words_bboxes)
    
    return so_prompt_phrase_word_box_list, overall_prompt, overall_phrases_words_bboxes
