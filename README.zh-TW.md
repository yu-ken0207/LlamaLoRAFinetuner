# Llama 模型訓練

[English](README.md) | **繁體中文**

[![Checks](https://github.com/yu-ken0207/LlamaLoRAFinetuner/actions/workflows/check.yml/badge.svg)](https://github.com/yu-ken0207/LlamaLoRAFinetuner/actions/workflows/check.yml)

**Built with Llama**：`outputs_*` 資料夾裡的 LoRA 權重由 Meta Llama 3.1 微調而來，授權見最下方。

## 概述

此程式碼包含一個完整的管道，用於訓練、微調及推理 Llama 模型。該管道包括以下組件：

1. **模型訓練** ：使用 LoRA（低秩適應）對 Llama 模型進行高效微調。
2. **數據集處理** ：靈活加載和預處理數據集，支持自定義的輸入與輸出格式。
3. **推理** ：使用已定義的提示模板生成模型預測。
4. **Hugging Face 集成** ：保存並將模型推送到 Hugging Face Hub。

---

## 功能特點

* **LoRA 微調** ：降低資源需求，實現高效微調。
* **靈活的數據集支持** ：使用可配置的鍵名，輕鬆適配不同格式的數據集。
* **基於指令學習** ：使用指令跟隨模板進行訓練和推理。
* **優化推理** ：使用 4-bit 量化和 FP16 加速推理。
* **Hugging Face Hub 集成** ：無縫保存和分享模型。

---

## 安裝與配置

### 先決條件

* Python 3.11+（實測 3.12）
* NVIDIA 顯示卡（CUDA）
* Conda（可選，用 venv 也可以）

### 安裝

1. 克隆此倉庫：
   ```bash
   git clone https://github.com/yu-ken0207/LlamaLoRAFinetuner.git
   cd LlamaLoRAFinetuner
   ```
2. 創建虛擬環境：
   ```bash
   conda env create -f environment.yml
   conda activate LlamaLoRAFinetuner
   ```
3. 依顯示卡的 CUDA 版本安裝 torch（其他 CUDA 版本的指令見 [PyTorch 官網](https://pytorch.org/get-started/locally/)），例如 CUDA 13.0：
   ```bash
   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
   ```
4. 安裝其他依賴：
   ```bash
   pip install -r requirements.txt
   ```

---

## 使用方法

### 訓練模型

使用自定義數據集訓練 Llama 模型：

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

訓練 repo 附的三個 Agent 用的指令在 `training_script.txt`。

### 推理

用 repo 附的權重推理（跟 `inference_test_TextStreamer.py` 相同；提示模板要跟訓練時一致）：

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

### 將模型保存到 Hugging Face Hub

確保將訓練好的模型保存並推送到 Hugging Face Hub：

```python
llama.save_model(local_path="outputs", hub_path="your-hf-username/test-model", token="your_hf_token", task_name="translation_task")
```

---

## 文件結構

* **`main.py`** ：訓練模型的入口點。
* **`llama_model.py`** ：處理模型的加載、微調和保存。
* **`dataset_loader.py`** ：管理數據集的加載與預處理。
* **`trainer_utils.py`** ：包含訓練工具和配置。
* **`environment.yml`** ：Conda 環境配置文件。
* **`requirements.txt`** ：使用 pip 的 Python 依賴列表。
* **`inference_test_TextStreamer.py`**、**`inference_test_apply_chat_template.py`** ：測試模型推理的腳本（兩種提示寫法）。
* **`training_script.txt`** ：三個 Agent 的訓練指令與資料範例。
* **`data_process/`** ：從 ALFWorld 資料整理出訓練資料的程式，以及整理好的 jsonl。
* **`outputs_*/`** ：訓練好的三個 LoRA 權重（任務拆解、子任務轉指令、動作分類）。

## 環境配置

`requirements.txt` 是 2026 年 11 月實測過的版本（torch 2.14.1、transformers 5.17.0、trl 1.15.0、unsloth 2026.10.11），實測環境是 Linux ARM64、NVIDIA Blackwell 顯示卡、CUDA 13.0、Python 3.12：

* 照 `main.py` 的流程訓練 30 步，loss 從 3.47 降到 0.37；訓練出的權重載入後能輸出正確的動作類別。
* 兩支 `inference_test_*.py` 載入 `outputs_*` 的權重推論正常。

x86 電腦照同樣步驟、把 torch 換成自己 CUDA 版本的安裝指令即可，但沒有實機測過；GitHub 自動檢查會確認套件版本在 x86、Python 3.11 和 3.12 都解得開。

已知還剩的漏洞：unsloth 目前只支援 datasets 5.0 以前的版本，datasets 4.8.5 與它限定版本的 fsspec 各有一個已知漏洞，分別出在「資料夾型資料集」與「Kerchunk 參考檔」，這個專案都沒有用到；等 unsloth 支援 datasets 5 再升級。

`data_process/` 的整理程式另外需要 ALFWorld 的環境，見 [ALFWorld](https://github.com/alfworld/alfworld) 的安裝說明。

`outputs_*` 的權重是 2025 年 1 月用 torch 2.5.1、transformers 4.47.1、unsloth 2024.12 訓練的，當時完整的套件清單在 git 歷史裡。

---

## 貢獻指南

1. Fork 此倉庫。
2. 創建您的功能分支：
   ```bash
   git checkout -b feature/new-feature
   ```
3. 提交您的更改：
   ```bash
   git commit -m "Add new feature"
   ```
4. 推送到分支：
   ```bash
   git push origin feature/new-feature
   ```
5. 開啟 Pull Request。

---

## License

- 程式碼：Apache-2.0，見 [LICENSE](LICENSE)。
- `data_process/alfworld/` 是 [ALFWorld](https://github.com/alfworld/alfworld) 的程式（MIT License，Copyright (c) 2020 Mohit Shridhar and others），授權見 [data_process/alfworld/LICENSE](data_process/alfworld/LICENSE)。
- `outputs_*` 資料夾裡的 LoRA 權重與 tokenizer：由 Meta Llama 3.1 衍生，依 [Llama 3.1 Community License](LICENSE-LLAMA-3.1) 散布，使用時要遵守 [Acceptable Use Policy](USE_POLICY-LLAMA-3.1.md)，另見 [NOTICE](NOTICE)。

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
