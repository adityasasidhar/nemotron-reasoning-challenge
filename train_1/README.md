---
base_model: /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
- lora
- transformers
---

# Nemotron-3-Nano-30B LoRA Adapter

Fine-tuned LoRA adapter for Nemotron-3-Nano-30B-A3B-BF16 on the NVIDIA Nemotron Model Reasoning Challenge.

## Model Details

- **Base Model:** Nemotron-3-Nano-30B-A3B-BF16
- **Model type:** LoRA Adapter
- **Language(s):** English
- **Finetuned from:** metric/nemotron-3-nano-30b-a3b-bf16

## Training Details

### Training Data

Supervised fine-tuning dataset with chain-of-thought reasoning traces for logical reasoning puzzles.

### Training Hyperparameters

- **Training regime:** bf16 mixed precision
- **LoRA rank:** 32
- **LoRA alpha:** 64
- **LoRA dropout:** 0.0
- **Learning rate:** 2e-4
- **LR scheduler:** Linear decay
- **Target modules:** in_proj, out_proj, q_proj, k_proj, v_proj, o_proj
- **Training steps:** 209
- **Batch size:** 32

### Framework versions

- PEFT 0.18.1