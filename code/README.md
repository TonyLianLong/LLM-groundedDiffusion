# Our code is available
Our demo code is available at [https://huggingface.co/spaces/longlian/llm-grounded-diffusion](https://huggingface.co/spaces/longlian/llm-grounded-diffusion).

This demo has stage 1 (text-to-layout) and stage 2 (layout-to-image). As stated in README of the demo, it differs slightly version from the paper (e.g., using frozen GLIGEN instead of backward guidance for faster generation), but the overall framework is the same.

## Trying our demo without installations
You can try it yourself on [HuggingFace space](https://huggingface.co/spaces/longlian/llm-grounded-diffusion).
You can find the source files that power the demo [Here](https://huggingface.co/spaces/longlian/llm-grounded-diffusion/tree/main).

## Running locally
If you want to benchmark or adapt the code locally, you can directly clone our repo to run locally and tweak the code:
Then:
```sh
cd code
cd llm-grounded-diffusion
# Install dependencies
pip install -r requirements.txt
# Optional: Remove caches for gradio examples if you tweak anything. The caches will be re-generated when you run the app for the first time.
rm -rf gradio_cached_examples
# Run our WebUI
python app.py
```

If you want to run your own generation in Python without using WebUI (e.g., batch generation), you can directly import the `run` function in `generation` (or `baseline`) without running `app.py`. See `app.py` for how it generates samples by importing `generation`.
