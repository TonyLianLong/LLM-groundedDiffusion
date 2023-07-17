---
title: LLM Grounded Diffusion
emoji: 😊
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 3.35.2
app_file: app.py
pinned: true
tags: [llm, diffusion, grounding, grounded, llm-grounded, text-to-image, language, large language models, layout, generation, generative, customization, personalization, prompting, chatgpt, gpt-3.5, gpt-4]
---

<h1>LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models</h1>
<h2>LLM + Stable Diffusion => better prompt understanding in text2image generation 🤩</h2>
<h2><a href='https://llm-grounded-diffusion.github.io/'>Project Page</a> | <a href='https://bair.berkeley.edu/blog/2023/05/23/lmd/'>5-minute Blog Post</a> | <a href='https://arxiv.org/pdf/2305.13655.pdf'>ArXiv Paper</a> (<a href='https://arxiv.org/abs/2305.13655'>ArXiv Abstract</a>) | <a href='https://github.com/TonyLianLong/LLM-groundedDiffusion'>Github</a> | <a href='https://llm-grounded-diffusion.github.io/#citation'>Cite our work</a> if our ideas inspire you.</h2>
<p><b>Tips:</b><p>
<p>1. If ChatGPT doesn't generate layout, add/remove the trailing space (added by default) and/or use GPT-4.</p>
<p>2. You can perform multi-round specification by giving ChatGPT follow-up requests (e.g., make the object boxes bigger).</p>
<p>3. You can also try prompts in Simplified Chinese. If you want to try prompts in another language, translate the first line of last example to your language.</p>
<p>4. The diffusion model only runs 20 steps by default. You can make it run 50 steps to get higher quality images (or tweak frozen steps/guidance steps for better guidance and coherence).</p>
<br/>
<p>Implementation note: In this demo, we replace the attention manipulation in our layout-guided Stable Diffusion described in our paper with GLIGEN due to much faster inference speed (<b>FlashAttention supported, no backprop needed</b> during inference). Compared to vanilla GLIGEN, we have better coherence. Other parts of text-to-image pipeline, including single object generation and SAM, remain the same. The settings and examples in the prompt are simplified in this demo.</p>

Credits:

This space uses code from [diffusers](https://huggingface.co/docs/diffusers/index), [GLIGEN](https://github.com/gligen/GLIGEN), and [layout-guidance](https://github.com/silent-chen/layout-guidance). Using their code means adhering to their license.
