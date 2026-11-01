# Llama LoRA Finetuner

**English** | [繁體中文](README.zh-TW.md)

[![Checks](https://github.com/yu-ken0207/LlamaLoRAFinetuner/actions/workflows/check.yml/badge.svg)](https://github.com/yu-ken0207/LlamaLoRAFinetuner/actions/workflows/check.yml)

A reusable pipeline for LoRA fine-tuning Llama 3.1 8B Instruct (4-bit, with Unsloth): load a Hugging Face dataset by naming its input and output fields, train with an instruction template, run inference, and push the result to the Hugging Face Hub. Used to train three small planning agents for household robot tasks from ALFRED / ALFWorld data: task decomposition, step-to-command planning, and action classification (adapters included in `outputs_*`).

**Built with Llama**: the LoRA weights in the `outputs_*` folders are fine-tuned from Meta Llama 3.1. See the license section at the bottom.

## Overview

The pipeline covers training, fine-tuning, and inference:

1. **Training**: efficient fine-tuning of Llama with LoRA (low-rank adaptation)
2. **Datasets**: load and preprocess datasets with configurable input and output fields
3. **Inference**: generate predictions with the same prompt template used in training
4. **Hugging Face integration**: save the model and push it to the Hugging Face Hub

## Features

* **LoRA fine-tuning**: lower resource needs
* **Flexible datasets**: name the input and output fields to fit different formats
* **Instruction tuning**: an instruction-following template for training and inference
* **Fast inference**: 4-bit quantization and FP16
* **Hugging Face Hub**: save and share models

## Installation

### Prerequisites

* Python 3.11+ (tested with 3.12)
* An NVIDIA GPU (CUDA)
* Conda (optional; venv also works)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yu-ken0207/LlamaLoRAFinetuner.git
   cd LlamaLoRAFinetuner
   ```
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate LlamaLoRAFinetuner
   ```
3. Install torch for your GPU's CUDA version (commands for other versions are on the [PyTorch website](https://pytorch.org/get-started/locally/)). For CUDA 13.0:
   ```bash
   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
   ```
4. Install the other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train

Train on your own dataset (this example translates modern Chinese into Classical Chinese):

```bash
python main.py \
  --model_name "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
  --dataset_name "AISPIN/shiji-70liezhuan" \
  --instruction "將現代漢語翻譯成文言文。" \
  --input_key "input" \
  --output_key "output" \
  --local_path "outputs" \
  --hub_path "your-hf-username/test-model" \
  --token "your_hf_token" \
  --task_name "translation_task"
```

The commands used to train the three included agents are in `training_script.txt`.

### Inference

Run an included adapter (same as `inference_test_TextStreamer.py`; the prompt template must match the one used in training):

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

def infer(model_path, instruction, input_text="", max_new_tokens=128):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        dtype="float16",
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input_text}

    ### Response:
    """

    inputs = tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens)

infer(
    model_path="outputs_TaskDecomposerAgent/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_TaskDecomposer_20250104_185454",
    instruction="以下是一個總體任務描述和其對應的高階步驟。請學習總體任務與高階步驟的關係，並根據高階步驟生成總體任務描述。",
    input_text="put some book on sidetable.",
)
```

The instructions are in Chinese because the agents were trained with Chinese instructions and English ALFWorld inputs.

### Push to the Hugging Face Hub

```python
llama.save_model(local_path="outputs", hub_path="your-hf-username/test-model", token="your_hf_token", task_name="translation_task")
```

## Files

* **`main.py`**: entry point for training
* **`llama_model.py`**: loads, fine-tunes, and saves the model
* **`dataset_loader.py`**: loads and preprocesses datasets
* **`trainer_utils.py`**: training settings
* **`environment.yml`**: Conda environment
* **`requirements.txt`**: Python dependencies for pip
* **`inference_test_TextStreamer.py`**, **`inference_test_apply_chat_template.py`**: inference test scripts (two prompt styles)
* **`training_script.txt`**: training commands and data samples for the three agents
* **`data_process/`**: scripts that build the training data from ALFWorld, plus the resulting jsonl files
* **`outputs_*/`**: the three trained LoRA adapters (task decomposition, sub-task to command, action classification)

## Tested environment

`requirements.txt` lists versions tested in November 2026 (torch 2.14.1, transformers 5.17.0, trl 1.15.0, unsloth 2026.10.11) on Linux ARM64 with an NVIDIA Blackwell GPU, CUDA 13.0, and Python 3.12:

* Training with the `main.py` flow for 30 steps brought the loss from 3.47 to 0.37, and the trained adapter outputs the correct action classes.
* Both `inference_test_*.py` scripts work with the adapters in `outputs_*`.

On x86, follow the same steps with the torch install command for your CUDA version; this was not tested on real hardware, but GitHub Actions checks that the dependency versions resolve on x86 with Python 3.11 and 3.12.

Remaining known vulnerabilities: unsloth currently supports only datasets versions before 5.0. datasets 4.8.5 and the fsspec version it pins each have one known vulnerability (in folder-based dataset builders and in Kerchunk reference files). This project uses neither. They will be fixed by upgrading once unsloth supports datasets 5.

The scripts in `data_process/` also need an ALFWorld environment; see the [ALFWorld](https://github.com/alfworld/alfworld) installation guide.

The adapters in `outputs_*` were trained in January 2025 with torch 2.5.1, transformers 4.47.1, and unsloth 2024.12. The full package list from that time is in the git history.

## Contributing

1. Fork this repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature/new-feature
   ```
5. Open a pull request.

## License

- Code: Apache-2.0, see [LICENSE](LICENSE).
- `data_process/alfworld/` is code from [ALFWorld](https://github.com/alfworld/alfworld) (MIT License, Copyright (c) 2020 Mohit Shridhar and others); see [data_process/alfworld/LICENSE](data_process/alfworld/LICENSE).
- The LoRA weights and tokenizers in `outputs_*`: derived from Meta Llama 3.1 and distributed under the [Llama 3.1 Community License](LICENSE-LLAMA-3.1). Use must follow the [Acceptable Use Policy](USE_POLICY-LLAMA-3.1.md); see also [NOTICE](NOTICE).

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
