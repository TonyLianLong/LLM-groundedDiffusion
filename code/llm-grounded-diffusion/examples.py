stage1_examples = [
    ["""A realistic photo of a wooden table with an apple on the left and a pear on the right."""],
    ["""A realistic photo of 4 TVs on a wall."""],
    ["""A realistic photo of a gray cat and an orange dog on the grass."""],
    ["""In an empty indoor scene, a blue cube directly above a red cube with a vase on the left of them."""],
    ["""A realistic photo of a wooden table without bananas in an indoor scene"""],
    ["""A realistic photo of two cars on the road."""],
    ["""一个室内场景的水彩画，一个桌子上面放着一盘水果"""]
]

# Layout, seed
stage2_examples = [
    ["""Caption: A realistic photo of a wooden table with an apple on the left and a pear on the right.
Objects: [('a wooden table', [30, 30, 452, 452]), ('an apple', [52, 223, 50, 60]), ('a pear', [400, 240, 50, 60])]
Background prompt: A realistic photo""", "", 0],
    ["""Caption: A realistic photo of 4 TVs on a wall.
Objects: [('a TV', [12, 108, 120, 100]), ('a TV', [132, 112, 120, 100]), ('a TV', [252, 104, 120, 100]), ('a TV', [372, 106, 120, 100])]
Background prompt: A realistic photo of a wall""", "", 0],
    ["""Caption: A realistic photo of a gray cat and an orange dog on the grass.
Objects: [('a gray cat', [67, 243, 120, 126]), ('an orange dog', [265, 193, 190, 210])]
Background prompt: A realistic photo of a grassy area.""", "", 0],
    ["""Caption: 一个室内场景的水彩画，一个桌子上面放着一盘水果
Objects: [('a table', [81, 242, 350, 210]), ('a plate of fruits', [151, 287, 210, 117])]
Background prompt: A watercolor painting of an indoor scene""", "", 1],
    ["""Caption: In an empty indoor scene, a blue cube directly above a red cube with a vase on the left of them.
Objects: [('a blue cube', [232, 116, 76, 76]), ('a red cube', [232, 212, 76, 76]), ('a vase', [100, 198, 62, 144])]
Background prompt: An empty indoor scene""", "", 2],
    ["""Caption: A realistic photo of a wooden table without bananas in an indoor scene
Objects: [('a wooden table', [75, 256, 365, 156])]
Background prompt: A realistic photo of an indoor scene""", "", 3],
    ["""Caption: A realistic photo of two cars on the road.
Objects: [('a car', [20, 242, 235, 185]), ('a car', [275, 246, 215, 180])]
Background prompt: A realistic photo of a road.""", "A realistic photo of two cars on the road.", 4],
]
