---
base_model: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
library_name: peft
license: llama3.1
---

# Llama-3.1-8B-Instruct-bnb-4bit_TaskDecomposer

LoRA adapter for Meta Llama 3.1 8B Instruct (4-bit, loaded with Unsloth). Given a household task (for example "Put a clean ladle on the kitchen island"), it lists the high-level steps to complete it.

**Built with Llama**

- Base model: [unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)
- Training data: task, step and action descriptions extracted from the ALFRED / ALFWorld household tasks (MIT License)
- Training code: [LlamaLoRAFinetuner](https://github.com/yu-ken0207/LlamaLoRAFinetuner)
- License: Llama 3.1 Community License ([LICENSE-LLAMA-3.1](../../../LICENSE-LLAMA-3.1)); use must follow the [Llama 3.1 Acceptable Use Policy](../../../USE_POLICY-LLAMA-3.1.md)

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
