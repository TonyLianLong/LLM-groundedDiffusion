# LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models
[Long Lian](https://tonylian.com/), [Boyi Li](https://sites.google.com/site/boyilics/home), [Adam Yala](https://www.adamyala.org/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) at UC Berkeley/UCSF.

[Paper](https://arxiv.org/pdf/2305.13655.pdf) | [Project Page](https://llm-grounded-diffusion.github.io/) | [**5-minute Blog Post**](https://bair.berkeley.edu/blog/2023/05/23/lmd/) | [**HuggingFace Demo (stage 1 and 2)**](https://huggingface.co/spaces/longlian/llm-grounded-diffusion) | [Citation](#citation)

**TL;DR**: Text Prompt -> LLM as a request parser -> Intermediate Representation (such as an image layout) -> Stable Diffusion -> Image.

![Main Image](https://llm-grounded-diffusion.github.io/main_figure.jpg)

## Updates
**Our huggingface demo for stage 1 and 2 is released! [Check it out here](https://huggingface.co/spaces/longlian/llm-grounded-diffusion).**

**Our code (stage 1 and stage 2) is also available to run locally. [The code and instructions to run](code/README.md).**

## Our code (with Web UI)
Our code that supports text-to-layout (stage 1) and layout-to-image (stage 2) generation is released. [Click here to see the code and instructions to run](code/README.md). 

## LLM-grounded Diffusion (LMD)
### Enhanced Prompt Understanding
![Visualizations](https://llm-grounded-diffusion.github.io/visualizations.jpg)

### Additional Capabilities: Multi-round Scene Specification/Generation from Non-English Prompts
![Additional Capabilities](https://llm-grounded-diffusion.github.io/additional_abilities.jpg)
![Additional Capabilities GIF](https://llm-grounded-diffusion.github.io/multiround.gif)

## Contact us
Please contact Long (Tony) Lian if you have any questions: `longlian@berkeley.edu`.

## Citation
If you use this work or find it helpful, please consider giving a citation.
```
@article{lian2023llmgrounded,
    title={LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models}, 
    author={Lian, Long and Li, Boyi and Yala, Adam and Darrell, Trevor},
    journal={arXiv preprint arXiv:2305.13655},
    year={2023}
}
```
