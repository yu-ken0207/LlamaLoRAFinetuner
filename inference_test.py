from unsloth import FastLanguageModel
from transformers import TextStreamer

def infer(model_path, instruction, input_text="", max_new_tokens=128):
    # 加載模型和分詞器
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,  # 調整序列長度
        dtype="float16",  # 使用 FP16
        load_in_4bit=True,  # 加載 4-bit 模型（若適用）
    )

    # 啟用推理加速
    FastLanguageModel.for_inference(model)

    # 定義提示模板（與訓練一致）
    alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input_text}

    ### Response:
    """

    # 將輸入文本轉換為張量
    inputs = tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")

    # 使用 TextStreamer 進行實時文本生成
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    # 模型保存的路徑
    model_path = "outputs_TaskDecomposerAgent/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_TaskDecomposer_20250104_185454"

    # 指令和輸入
    instruction = "以下是一個總體任務描述和其對應的高階步驟。請學習總體任務與高階步驟的關係，並根據高階步驟生成總體任務描述。"
    input_text = "Hold the clock and turn on the lamp."

    # 執行推理
    infer(model_path, instruction, input_text)
