# Our demo code is available
Our demo code is available at [https://huggingface.co/spaces/longlian/llm-grounded-diffusion](https://huggingface.co/spaces/longlian/llm-grounded-diffusion).

This demo has stage 1 (text-to-layout) and stage 2 (layout-to-image). As stated in README of the demo, it differs slightly version from the paper (e.g., using simplified examples), but the overall framework is the same.

We will release the benchmark in the main Github repo.

You can run it yourself on [HuggingFace space](https://huggingface.co/spaces/longlian/llm-grounded-diffusion).

You can find the source files that power the demo [Here](https://huggingface.co/spaces/longlian/llm-grounded-diffusion/tree/main).

You can directly clone the demo to run locally and tweak the code (remember to remove the cache):
```
git clone https://huggingface.co/spaces/longlian/llm-grounded-diffusion
pip install -r requirements.txt
python app.py
```
