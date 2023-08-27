# Our WebUI is available
In addition to using our demo on [HuggingFace space](https://huggingface.co/spaces/longlian/llm-grounded-diffusion), you can check the source code of our demo WebUI code at [https://huggingface.co/spaces/longlian/llm-grounded-diffusion](https://huggingface.co/spaces/longlian/llm-grounded-diffusion).

This demo has stage 1 (text-to-layout) and stage 2 (layout-to-image).

The updated demo provides different presets for different generation purpose. The full version is LMD+. You can enable/disable each type of guidance for ablation.

~~As stated in README of the demo, it differs slightly version from the paper (e.g., using frozen GLIGEN instead of backward guidance for faster generation), but the overall framework is the same.~~

## Trying our demo without installations
You can try it yourself on [HuggingFace space](https://huggingface.co/spaces/longlian/llm-grounded-diffusion).

## Running locally (likely much faster with a recent GPU)
If you want to benchmark or adapt the code locally, you can directly clone our repo to run locally and tweak the code:
Then:
```sh
git clone https://huggingface.co/spaces/longlian/llm-grounded-diffusion
cd llm-grounded-diffusion
# Install dependencies
pip install -r requirements.txt
# Optional: Remove caches for gradio examples if you tweak anything. The caches will be re-generated when you run the app for the first time.
# rm -rf gradio_cached_examples
# Run our WebUI
python app.py
```
