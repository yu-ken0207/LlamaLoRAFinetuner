# Llama 模型訓練

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

* Python 3.10+
* Conda（可選，但推薦使用）

### 安裝

1. 克隆此倉庫：
   ```bash
   git clone https://github.com/LlamaLoRAFinetuner.git
   cd LlamaLoRAFinetuner
   ```
2. 創建虛擬環境並安裝依賴：
   ```bash
   conda env create -f environment.yml
   conda activate train_llama
   ```
3. 安裝額外的 Python 依賴（如有需要）：
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

### 推理

使用已訓練的模型進行推理：

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

    alpaca_prompt = f"""以下是一個描述任務的指令，並附有提供進一步上下文的輸入內容。請撰寫適當的回應完成此請求。

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
    instruction="將以下任務轉換為逐步可執行的行動。",
    input_text="清理桌面並整理書籍。"
)
```

### 將模型保存到 Hugging Face Hub

確保將訓練好的模型保存並推送到 Hugging Face Hub：

```python
llama.save_model(local_path="outputs", hub_path="your-hf-username/test-model", token="your_hf_token", task_name="translation_task")
```

---

---

## 文件結構

* **`main.py`** ：訓練模型的入口點。
* **`llama_model.py`** ：處理模型的加載、微調和保存。
* **`dataset_loader.py`** ：管理數據集的加載與預處理。
* **`trainer_utils.py`** ：包含訓練工具和配置。
* **`environment.yml`** ：Conda 環境配置文件。
* **`requirements.txt`** ：使用 pip 的 Python 依賴列表。
* **`inference_test.py`** ：用於測試模型推理的腳本。

## 環境配置

若需重建環境，可使用提供的 `environment.yml` 文件：

```bash
conda env create -f environment.yml
```

或使用 `requirements.txt`：

```bash
pip install -r requirements.txt
```

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

## 授權

此項目基於 Apache License 2.0 授權。詳細信息請參閱 `LICENSE` 文件。
