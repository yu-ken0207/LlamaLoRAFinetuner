from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch


def infer(model_path, instruction, input_text="", max_new_tokens=128):
    # 加載模型和 tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,  # 設定最大序列長度
        dtype="float16",  # 使用 FP16
        load_in_4bit=True,  # 加載 4-bit 量化模型
    )

    # 啟用推理模式
    FastLanguageModel.for_inference(model)

    # 使用 ChatML 格式來構造對話
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"{instruction}\n\n{input_text}"}
    ]

    # 使用 apply_chat_template 來轉換輸入為 Chat 格式
    inputs = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,  # 自動添加模型需要的提示
        return_dict=True,
        return_tensors="pt"  # 轉換為 PyTorch 張量
    )

    # 確保張量在模型所在設備上
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # 生成文本
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,  # 控制隨機性
        top_p=0.9,  # 取前 90% 的概率質量
        repetition_penalty=1.1,  # 避免重複
    )

    # 提取生成的文本（去掉輸入部分，只保留輸出）
    generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()

    # 印出生成的內容
    print(generated_text)

if __name__ == "__main__":

    # model_path = "outputs_TaskDecomposerAgent/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_TaskDecomposer_20250104_185454"
    # instruction = "以下是一個總體任務描述和其對應的高階步驟。請學習總體任務與高階步驟的關係，並根據高階步驟生成總體任務描述。"
    # input_text = "Hold the clock and turn on the lamp."
    # infer(model_path, instruction, input_text)

    # model_path = "outputs_TaskActionClassifierAgent/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_TaskActionClassifier_20250104_184343"
    # instruction = "請閱讀高階描述（high_desc），並執行相應的行為（action），如移動到某個位置或拾取物品"
    # input_text = "Turn right and walk to the dresser."
    # infer(model_path, instruction, input_text)

    model_path = "outputs_SubTaskPlannerAgent/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_SubTaskPlanner_20250104_183624"
    instruction = "你是一個室內導航助手。請根據子任務描述（subTask_descs），決定下一步需要執行的動作（action）"
    input_text = "Pick the book up from the cabinet."
    infer(model_path, instruction, input_text)