# LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models
[Long Lian](https://tonylian.com/), [Boyi Li](https://sites.google.com/site/boyilics/home), [Adam Yala](https://www.adamyala.org/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) at UC Berkeley/UCSF.

[Paper](https://arxiv.org/pdf/2305.13655.pdf) | [Project Page](https://llm-grounded-diffusion.github.io/) | [**5-minute Blog Post**](https://bair.berkeley.edu/blog/2023/05/23/lmd/) | [**HuggingFace Demo (updated!)**](https://huggingface.co/spaces/longlian/llm-grounded-diffusion) | [Citation](#citation)

**TL;DR**: Text Prompt -> LLM as a Request Parser -> Intermediate Representation (such as an image layout) -> Stable Diffusion -> Image.

![Main Image](https://llm-grounded-diffusion.github.io/main_figure.jpg)
![Visualizations: Enhanced Prompt Understanding](https://llm-grounded-diffusion.github.io/visualizations.jpg)

## Updates
**Our repo has been largely improved: now we have a repo with many methods implemented, including our training-free LMD and LMD+ (LMD with GLIGEN adapters).**

**Our huggingface WebUI demo for stage 1 and 2 is updated: now we support enabling each of the guidance components to get a taste of contributions! [Check it out here](https://huggingface.co/spaces/longlian/llm-grounded-diffusion).**

Our WebUI is also available to run locally. [The instructions to run our WebUI locally to get faster generation without queues are here](webui/README.md).

## Our repo implements the following layout-to-image methods (stage 2)
These methods can be freely combined with our proposed LLM-based box-to-layout method (stage 1) also implemented in this repo.

- [x] Training-Free LMD (Using original SD v1/v2 weights)
- [x] LMD+ (Training-Free LMD that uses both cross-attention control and GLIGEN adapters) 
- [x] [Layout Guidance (Backward Guidance)](https://arxiv.org/abs/2304.03373)
- [x] [BoxDiff (ICCV '23)](https://arxiv.org/abs/2307.10816)
- [x] [MultiDiffusion (Region Control, ICML '23)](https://arxiv.org/abs/2302.08113)
- [x] [GLIGEN (CVPR '23)](https://arxiv.org/abs/2301.07093)

Feel free to contact me / submit a pull request to add your methods!

## Our repo's features
* **Both web-based ChatGPT and OpenAI API on GPT-3.5/4 supported**: Allows generating bounding boxes by either asking ChatGPT yourself (free) or in batch with OpenAI API (fully automated).
* **LLM queries are cached to save $$$ on LLM APIs:** we cache each LLM query for layout generation so it does not re-generate the layouts from the same prompt.
* **Open-source LLMs supported!**: Host LLMs yourself for more freedom and lower costs! We support Vicuna, LLaMA 2, StableBeluga2, etc. More in [FAQ](#FAQ).
* **Supports both LMD** (which uses SD weights without training and performs attention guidance) **and LMD+** (which adds GLIGEN adapters to SD in addition to attention guidance)
* **Supports SD v1 and SD v2 in the same codebase**: if you implement a new feature or a new loss, it's likely that it will work on both SD v1 and v2.
* **Several baseline stage 2 methods implemented in the same codebase**: handy if you want to benchmark and compare
* **Hackable**: we provide a minimal copy of diffusion UNet architecture in our repo that exports the attention maps according to your need. This allows you to change things **without maintaining your own diffusers package**.
* **Parallel and resumable image generation supported!** You can generate in parallel to make use of multiple GPUs/servers. If a generation fails on some images (e.g., CUDA OOM), you can simply rerun generation to regenerate those. More in [FAQ](FAQ).
* **Modular**: we implement different methods in different files. Copy from a file in `generation` and start creating your method without impacting existing methods.
* **Web UI supported**: don't want to code or run anything? Try our [public WebUI demo](https://huggingface.co/spaces/longlian/llm-grounded-diffusion) or instructions to [run WebUI locally](webui/README.md).

<details>
<summary> And more exciting features! Expand to see. </summary>

* **FlashAttention and PyTorch v2 supported**.
* **Unified benchmark:** same evaluation protocol on layouts (stage 1) and generated images (stage 1+2) for all methods implemented. The benchmark is in beta and might change.
* **Provides different presets** to balance better control and fast generation in Web UI.

</details>

# LLM-grounded Diffusion (LMD)
We provide instructions to run our code in this section.
## Installation
```
pip install -r requirements.txt
```

## Stage 1: Text-to-Layout Generation
**Note that we have uploaded the layout caches into this repo so that you can skip this step if you don't need layouts for new prompts.**

Since we have cached the layout generation (which will be downloaded when you clone the repo), **you need to remove the cache in `cache` directory if you want to re-generate the layout with the same prompts**.

**Our layout generation format:** The LLM takes in a text prompt describing the image and outputs three elements: **1.** captioned boxes, **2.** a background prompt, **3.** a negative prompt (useful if the LLM wants to express negation). The template and examples are in [prompt.py](https://github.com/TonyLianLong/LLM-groundedDiffusion/blob/main/prompt.py). You can edit the template and the parsing function to ask the LLM to generate additional things or even perform chain-of-thought for better generation.

### Option 1 (automated): Use an OpenAI API key
If you have an [OpenAI API key](https://openai.com/blog/openai-api), you can put the API key in `utils/api_key.py` or set `OPENAI_API_KEY` environment variable. Then you can use OpenAI's API for batch text-to-layout generation by querying an LLM, with GPT-4 as an example:
```
python prompt_batch.py --prompt-type demo --model gpt-4 --auto-query --always-save --template_version v0.1
```
`--prompt-type demo` includes a few prompts for demonstrations. The layout generation will be cached so it does not query the LLM again with the same prompt (lowers the cost).

You can visualize the bounding boxes in `img_generations/imgs_demo_templatev0.1`.

### Option 2 (free): Manually copy and paste to ChatGPT
```
python prompt_batch.py --prompt-type demo --model gpt-4 --always-save --template_version v0.1
```
Then copy and paste the template to [ChatGPT](https://chat.openai.com). Note that you want to use GPT-4 or change the `--model` to gpt-3.5 in order to match the cache file name. Then copy the response back. The generation will be cached.

If you want to visualize before deciding to save or not, you don't need to pass in `--always-save`.

### Run our benchmark on text-to-layout generation evaluation
We provide a benchmark that applies both to stage 1 and stage 2. This benchmarks includes a set of prompts with four tasks (negation, numeracy, attribute binding, and spatial relationships) as well as unified benchmarking code for all implemented methods and both stages.

This will generate layouts from the prompts in the benchmark (with `--prompt-type lmd`) and evaluate the results:
```
python prompt_batch.py --prompt-type lmd --model gpt-3.5 --auto-query --always-save --template_version v0.1
python scripts/eval_stage_one.py --prompt-type lmd --model gpt-3.5 --template_version v0.1
```
<details>
  <summary>Our reference benchmark results (stage 1, evaluating the generated layouts only)</summary>

| Method  | Negation | Numeracy | Attribution | Spatial | Overall    |
| ------- | -------- | -------- | ----------- | ------- | ---------- |
| GPT-3.5 | 100      | 97       | 100         | 99      | 99.0%      |
| GPT-4   | 100      | 100      | 100         | 100     | **100.0%** |

<!-- * GPT-3.5:

```
Eval type: negation, success: 100/100, rate: 1.00
Eval type: numeracy, success: 97/100, rate: 0.97
Eval type: attribution, success: 100/100, rate: 1.00
Eval type: spatial, success: 99/100, rate: 0.99
Overall: success: 396/400, rate: 0.99
```

* GPT-4:

```
Eval type: negation, success: 100/100, rate: 1.00
Eval type: numeracy, success: 100/100, rate: 1.00
Eval type: attribution, success: 100/100, rate: 1.00
Eval type: spatial, success: 100/100, rate: 1.00
Overall: success: 400/400, rate: 1.00
``` -->
</details>

## Stage 2: Layout-to-Image Generation
Note that since we provide caches for stage 1, you don't need to run stage 1 on your own for cached prompts that we provide (i.e., you don't need an OpenAI API key or to query an LLM).

Run layout-to-image generation using the gpt-4 cache and LMD+:
```
python generate.py --prompt-type demo --model gpt-4 --save-suffix "gpt-4" --repeats 5 --frozen_step_ratio 0.5 --regenerate 1 --force_run_ind 0 --run-model lmd_plus --no-scale-boxes-default --template_version v0.1
```

`--save-suffix` is the suffix added to the name of the run. You can change that if you change the args to mark the setting in the runs. `--run-model` specifies the method to run. You can set to LMD/LMD+ or the implemented baselines (with examples below). Use `--use-sdv2` to enable SDv2.

### Run our benchmark on layout-to-image generation evaluation
We use a unified evaluation metric as stage 1 in stage 2 (`--prompt-type lmd`). Since we have layout boxes for stage 1 but only images for stage 2, we use OWL-ViT in order to detect the objects and ensure they are generated (or not generated in negation) in the right number, with the right attributes, and in the right place. This benchmark is still in beta stage.

This runs generation with LMD+ and evaluate the generation: 
```shell
# Use GPT-3.5 layouts
python generate.py --prompt-type lmd --model gpt-3.5 --save-suffix "gpt-3.5" --repeats 1 --frozen_step_ratio 0.5 --regenerate 1 --force_run_ind 0 --run-model lmd_plus --no-scale-boxes-default --template_version v0.1
python scripts/owl_vit_eval.py --model gpt-3.5 --run_base_path img_generations/img_generations_templatev0.1_lmd_plus_lmd_gpt-3.5/run0 --skip_first_prompts 0 --prompt_start_ind 0 --verbose --detection_score_threshold 0.15 --nms_threshold 0.15 --class-aware-nms
# Use GPT-4 layouts
python generate.py --prompt-type lmd --model gpt-4 --save-suffix "gpt-4" --repeats 1 --frozen_step_ratio 0.5 --regenerate 1 --force_run_ind 0 --run-model lmd_plus --no-scale-boxes-default --template_version v0.1
python scripts/owl_vit_eval.py --model gpt-4 --run_base_path img_generations/img_generations_templatev0.1_lmd_plus_lmd_gpt-4/run0 --skip_first_prompts 0 --prompt_start_ind 0 --verbose --detection_score_threshold 0.15 --nms_threshold 0.15 --class-aware-nms
```

##### Our reference benchmark results
| Method                | Negation | Numeracy | Attribution | Spatial | Overall   |
| --------------------- | -------- | -------- | ----------- | ------- | --------- |
| SD v1.5               | 28       | 39       | 52          | 28      | 36.8%     |
| LMD+ (GPT-3.5)        | 100      | 86       | 69          | 67      | 80.5%     |
| LMD+ (GPT-4)          | 100      | 84       | 79          | 82      | **86.3%** |
| LMD+ (StableBeluga2*) | 88       | 60       | 56          | 64      | 67.0%     |

\* [StableBeluga2](https://huggingface.co/stabilityai/StableBeluga2) is an open-sourced model based on Llama 2. We discover that the fact that LLMs' spatial reasoning ability is also applicable to open-sourced models. However, it can still be improved, compared to proprietary models. We leave LLM fine-tuning for better layout generation in stage 1 to future research.

To run generation with LMD with original SD weights and evaluate the generation:
<details>
  <summary>Generate and evaluate samples with LMD</summary>

```shell
# Use GPT-3.5 layouts
python generate.py --prompt-type lmd --model gpt-3.5 --save-suffix "gpt-3.5" --repeats 1 --frozen_step_ratio 0.5 --regenerate 1 --force_run_ind 0 --run-model lmd --no-scale-boxes-default --template_version v0.1
python scripts/owl_vit_eval.py --model gpt-3.5 --run_base_path img_generations/img_generations_templatev0.1_lmd_lmd_gpt-3.5/run0 --skip_first_prompts 0 --prompt_start_ind 0 --verbose --detection_score_threshold 0.15 --nms_threshold 0.15 --class-aware-nms
```

  Note: You can enable autocast (mixed precision) to reduce the memory used in generation with `--use_autocast 1` with potentially slightly lower generation quality.
</details>

<details>
  <summary>Generate samples with other stage 2 baseline methods</summary>

```shell
# SD v1.5
python generate.py --prompt-type lmd --model gpt-3.5 --save-suffix "gpt-3.5" --repeats 1 --regenerate 1 --force_run_ind 0 --run-model sd --no-scale-boxes-default --template_version v0.1 --ignore-negative-prompt
# MultiDiffusion (training-free)
python generate.py --prompt-type lmd --model gpt-3.5 --save-suffix "gpt-3.5" --repeats 1 --regenerate 1 --force_run_ind 0 --run-model multidiffusion --no-scale-boxes-default --template_version v0.1 --multidiffusion_bootstrapping 10
# Backward Guidance (training-free)
python generate.py --prompt-type lmd --model gpt-3.5 --save-suffix "gpt-3.5" --repeats 1 --regenerate 1 --force_run_ind 0 --run-model backward_guidance --no-scale-boxes-default --template_version v0.1
# Boxdiff (training-free, our reimplementation)
python generate.py --prompt-type lmd --model gpt-3.5 --save-suffix "gpt-3.5" --repeats 1 --regenerate 1 --force_run_ind 0 --run-model boxdiff --no-scale-boxes-default --template_version v0.1
# GLIGEN (training-based)
python generate.py --prompt-type lmd --model gpt-3.5 --save-suffix "gpt-3.5" --repeats 1 --regenerate 1 --force_run_ind 0 --run-model gligen --no-scale-boxes-default --template_version v0.1
```

Note: we set `--ignore-negative-prompt` in SD v1.5 so that SD generation does not depend on the LLM and follows a text-to-image generation baseline (otherwise we take the LLM-generated negative prompts and put them into the negative prompt). For other baselines, you can feel free to generate. Evaluation is similar to LMD+, except you need to change the image path in the evaluation command.
</details>

<details>
  <summary>Our reference benchmark results (stage 2, LMD, without autocast)</summary>

| Method        | Negation | Numeracy | Attribution | Spatial | Overall |
| ------------- | -------- | -------- | ----------- | ------- | ------- |
| SD v1.5       | 28       | 39       | 52          | 28      | 36.8%   |
| LMD (GPT-3.5) | 100      | 62       | 65          | 79      | 76.5%   |
</details>

##### Ablation: Our reference benchmark results by combining LMD stage 1 with various layout-to-image baselines as stage 2</summary>
The stage 1 in this table is LMD (GPT-3.5) unless stated otherwise. We keep stage 1 in LMD the same and replace the stage 2 by other layout-to-image methods.

| Stage 1 / Stage 2 Method                                | Negation* | Numeracy | Attribution | Spatial | Overall   |
| ------------------------------------------------------- | --------- | -------- | ----------- | ------- | --------- |
| None / SD v1.5                                          | 28        | 39       | 52          | 28      | 36.8%     |
| _Training-free: <br/> (uses SD weights out-of-the-box)_ |
| **LMD** / MultiDiffusion                                | 100       | 30       | 42          | 36      | 52.0%     |
| **LMD** / Backward Guidance                             | 100       | 42       | 36          | 61      | 59.8%     |
| **LMD** / BoxDiff                                       | 100       | 32       | 55          | 62      | 62.3%     |
| **LMD** / **LMD**                                       | 100       | 62       | 65          | 79      | **76.5%** |
| _Training-based:_                                       |
| **LMD** / GLIGEN                                        | 100       | 57       | 57          | 45      | 64.8%     |
| **LMD** / **LMD+**\*\*                                  | 100       | 86       | 69          | 67      | 80.5%     |
| **LMD** / **LMD+** (GPT-4)                              | 100       | 84       | 79          | 82      | **86.3%** |

\* All methods equipped with LMD stage 1 understand negation well because LMD stage 1 generates the negative prompts, which is applicable to all methods that use classifier-free guidance on SD.

\*\* Note that LMD+ uses attention control that we proposed **in addition to** GLIGEN, which has much better generation compared to using only GLIGEN, showing that **our proposed training-free control is orthogonal to training-based methods such as GLIGEN**.

## FAQs
### How do I use open-source LLMs (e.g., LLaMA-2, StableBeluga2, Vicuna)?
You can install [fastchat](https://github.com/lm-sys/FastChat) and start a LLM server (note that the server does not have to be the same one as this repo). Using StableBeluga2 as an example (which performs the best among all open-source LLMs from our experience):

```shell
pip install fschat

export FASTCHAT_WORKER_API_TIMEOUT=600
python3 -m fastchat.serve.controller
CUDA_VISIBLE_DEVICES=0,1 python3 -m fastchat.serve.cli --model-path stabilityai/StableBeluga2 --num-gpus 2
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```
StableBeluga2 is a 70b model so you need at least 2 GPUs, but you can run smaller models with only 1 GPU. Simply replace the model path to the huggingface model key (e.g., `meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-33b-v1.3`). Note that you probably want models without RLHF (e.g., not `Llama-2-70b-chat-hf`), as we use text completion endpoints for layout generation. Then change the `--model` argument to the intended model.

If your LLM server is not on `localhost:8000`, you can change the API endpoint URL in [utils/llm.py](https://github.com/TonyLianLong/LLM-groundedDiffusion/blob/main/utils/llm.py). If you model name is not in the list in [utils/llm.py](https://github.com/TonyLianLong/LLM-groundedDiffusion/blob/main/utils/llm.py), you can add it to the `model_names` list. We created this list to prevent typos in the command.

### My LLM queries finished very quickly, why?
Check whether you have a lot of cache hits in the output. If so, you might want to use the cache (you are all set) or remove the cache in `cache` directory to regenerate.

Note that we allows different versions of templates so that you can manage several templates easily without cache overwrites.

## Contact us
Please contact Long (Tony) Lian if you have any questions: `longlian@berkeley.edu`.

## Acknowledgements
This repo uses code from [diffusers](https://huggingface.co/docs/diffusers/index), [GLIGEN](https://github.com/gligen/GLIGEN), and [layout-guidance](https://github.com/silent-chen/layout-guidance). This code also has an implementation of [boxdiff](https://github.com/showlab/BoxDiff) and [MultiDiffusion (region control)](https://github.com/omerbt/MultiDiffusion/tree/master). Using their code means adhering to their license.

## Citation
If you use our work or our implementation in this repo, or find them helpful, please consider giving a citation.
```
@article{lian2023llmgrounded,
    title={LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models}, 
    author={Lian, Long and Li, Boyi and Yala, Adam and Darrell, Trevor},
    journal={arXiv preprint arXiv:2305.13655},
    year={2023}
}
```
