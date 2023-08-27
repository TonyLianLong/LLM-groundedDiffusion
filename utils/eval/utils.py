import numpy as np
import inflect
import re

p = inflect.engine()


# Credit: GPT
def find_word_after(text, word):
    pattern = r"\b" + re.escape(word) + r"\s+(.+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


word_to_num_mapping = {p.number_to_words(i): i for i in range(1, 21)}

# New predicates that use the center
locations_xyxy = {
    ('left', 'right'): (lambda box1, box2: (box1[0] + box1[2]) < (box2[0] + box2[2])),
    ('right', 'left'): (lambda box1, box2: (box1[0] + box1[2]) > (box2[0] + box2[2])),
    ('top', 'bottom'): (lambda box1, box2: (box1[1] + box1[3]) < (box2[1] + box2[3])),
    ('bottom', 'top'): (lambda box1, box2: (box1[1] + box1[3]) > (box2[1] + box2[3]))
}

locations_xywh = {
    ('left', 'right'): (lambda box1, box2: box1[0] + box1[2]/2 < box2[0] + box2[2]/2),
    ('right', 'left'): (lambda box1, box2: box1[0] + box1[2]/2 > box2[0] + box2[2]/2),
    ('top', 'bottom'): (lambda box1, box2: box1[1] + box1[3]/2 < box2[1] + box2[3]/2),
    ('bottom', 'top'): (lambda box1, box2: box1[1] + box1[3]/2 > box2[1] + box2[3]/2)
}


def singular(noun):
    singular_noun = p.singular_noun(noun)
    if singular_noun is False:
        return noun
    return singular_noun


def get_box(gen_boxes, name_include):
    # This prevents substring match on non-word boundaries: carrot vs car
    box_match = [any([((name_include_item + ' ') in box['name'] or box['name'].endswith(name_include_item))
                     for name_include_item in name_include]) for box in gen_boxes]

    if not any(box_match):
        return None

    box_ind = np.min(np.where(box_match)[0])
    return gen_boxes[box_ind]


def count(gen_boxes, name_include):
    return sum([
        any([name_include_item in box['name'] for name_include_item in name_include]) for box in gen_boxes
    ])


def predicate_numeracy(query_names, intended_count, gen_boxes, verbose=False):
    # gen_boxes: dict with keys 'name' and 'bounding_box'
    object_count = count(gen_boxes, name_include=query_names)
    if verbose:
        print(
            f"object_count: {object_count}, intended_count: {intended_count} (gen_boxes: {gen_boxes}, query_names: {query_names})")

    return object_count == intended_count

def predicate_numeracy_2obj(query_names1, intended_count1, query_names2, intended_count2, gen_boxes, verbose=False):
    # gen_boxes: dict with keys 'name' and 'bounding_box'
    object_count1 = count(gen_boxes, name_include=query_names1)
    object_count2 = count(gen_boxes, name_include=query_names2)

    if verbose:
        print(
            f"object_count1: {object_count1}, intended_count1: {intended_count1} (gen_boxes: {gen_boxes}, query_names1: {query_names1})")
        print(
            f"object_count2: {object_count2}, intended_count2: {intended_count2} (gen_boxes: {gen_boxes}, query_names2: {query_names2})")

    return object_count1 == intended_count1 and object_count2 == intended_count2


def predicate_attribution(query_names1, query_names2, modifier1, modifier2, intended_count1, intended_count2, gen_boxes, verbose=False):
    # gen_boxes: dict with keys 'name' and 'bounding_box'
    if modifier1:
        query_names1 = [f"{modifier1} {item}" for item in query_names1]
    object_count1 = count(gen_boxes, name_include=query_names1)
    
    if query_names2 is not None:
        if modifier2:
            query_names2 = [f"{modifier2} {item}" for item in query_names2]
        object_count2 = count(gen_boxes, name_include=query_names2)

        if verbose:
            print(f"Count 1: {object_count1}, Count 2: {object_count2}")
        return object_count1 >= intended_count1 and object_count2 >= intended_count2
    else:
        if verbose:
            print(f"Count 1: {object_count1}")
        return object_count1 >= intended_count1    


def predicate_spatial(query_names1, query_names2, verify_fn, gen_boxes, verbose=False):
    # gen_boxes: dict with keys 'name' and 'bounding_box'

    object_box1 = get_box(gen_boxes, query_names1)
    object_box2 = get_box(gen_boxes, query_names2)

    if verbose:
        print(
            f"object_box1: {object_box1}, object_box2: {object_box2}")

    if object_box1 is None or object_box2 is None:
        return False

    return verify_fn(object_box1['bounding_box'], object_box2['bounding_box'])
