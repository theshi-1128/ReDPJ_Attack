# ReDPJ

This repository contains the current implementation of **ReDPJ** (Reasoning-guided Dual-Path Jailbreak), including textual and visual attack paths with adaptive reasoning operators.


## Overview

This repository shares the code of our latest work on LLMs jailbreaking. In this work:

- We investigate an overlooked reasoning-safety attack surface by shifting the jailbreak paradigm from input-level obfuscation to reasoning-level manipulation. Our findings show that, even in the absence of explicit harmful intent, LLMs can reconstruct latent malicious intent through intermediate reasoning, revealing a critical alignment gap between output-level safety and reasoning-level safety.

- We present ReDPJ, a reasoning-guided dual-path jailbreak framework that steers the model's reasoning trajectory towards producing harmful content. ReDPJ introduces dual-modality attack paths and an adaptive reasoning-path guidance strategy, effectively exposing the intrinsic vulnerabilities in both textual and visual reasoning processes of LLMs.

- We perform extensive experiments on diverse LLMs, demonstrating ReDPJ's impressive performance in terms of attack effectiveness, efficiency, and transferability. In addition, we validate the effectiveness of key components in ReDPJ and analyze the reasons behind the success of ReDPJ.


## What Is Implemented

- Dual-path attack initiation:
  - Textual anchor construction
  - Visual anchor construction
- Adaptive reasoning-path guidance Strategy
    - Reasoning path adjustment
    - Anchor toxicity adjustment

## Project Structure

- `ReDPJ.py`: textual pipeline entry
- `ReDPJ_visual.py`: visual pipeline entry
- `pipeline/`: core attack workflow
- `llm/`: model/API wrappers
- `dataset`: default dataset

## Setup

```bash
conda create -n redpj python=3.11 -y
conda activate redpj
pip install -r requirements.txt
```

Before running, configure model access:

1. Set API fields in `llm/api_config.py`:
   - `api_key`
   - `base_url`
   - `model_name`
2. If you use local Hugging Face models, set paths in `llm/llm_model.py` (`MODEL_LIST`).

## Run Textual Path

```bash
python3 ReDPJ.py \
  --target_model deepseek_v3 \
  --assist_model glm4 \
  --judge_model gpt4o \
  --dataset_dir ./dataset/harmful_behaviors.csv \
  --max_attack_rounds 5 \
  --max_adjustment_rounds 5 \
  --target_model_cuda_id cuda:0 \
  --assist_model_cuda_id cuda:1 \
  --judge_model_cuda_id cuda:2
```

## Run Visual Path

```bash
python3 ReDPJ_visual.py \
  --target_model gpt4o_vl \
  --assist_model_text glm4 \
  --assist_model_img gpt_img \
  --judge_model gpt4o \
  --dataset_dir ./dataset/harmful_behaviors.csv \
  --max_attack_rounds 5 \
  --max_adjustment_rounds 5 \
  --target_model_cuda_id cuda:0 \
  --assist_model_cuda_id cuda:1 \
  --judge_model_cuda_id cuda:2
```

## Key Arguments

- `target_models`: comma-separated target model names. If set, it overrides `target_model` and `num_target_models`.
- `num_target_models`: number of target models to run in one pass.
- `target_model_1` ... `target_model_4`: optional per-model names when `num_target_models > 1`.
- `max_attack_rounds`: `Tpath` (reasoning-path adjustment budget)
- `max_adjustment_rounds`: `Tanchor` (anchor-adjustment budget)
- `save_interval`: periodic CSV save interval in seconds

## Output

- Textual results: `./output/ReDPJ_text/`
- Visual CSV results: `./output/ReDPJ_img/text/`
- Generated visual anchors: `./output/ReDPJ_img/img/`

Each record includes final response, label (`Ft`), query count, success round/step, and full reasoning trace (`trace`).
