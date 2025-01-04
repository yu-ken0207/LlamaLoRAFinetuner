import argparse
from llama_model import LlamaModel
from dataset_loader import DatasetLoader
from trainer_utils import train_model

def main():
    # 設定 argparse 並加入預設值
    parser = argparse.ArgumentParser(description="Train Llama Model with Dataset")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", 
        help="Name of the model (default: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="AISPIN/shiji-70liezhuan", 
        help="Name of the dataset (default: AISPIN/shiji-70liezhuan)"
    )
    parser.add_argument(
        "--instruction", 
        type=str, 
        default="請把現代漢語翻譯成古文", 
        help="Instruction for the task (default: 請把現代漢語翻譯成古文)"
    )
    parser.add_argument(
        "--input_key", 
        type=str, 
        default="input", 
        help="Key name in the dataset for the input text (default: input)"
    )
    parser.add_argument(
        "--output_key", 
        type=str, 
        default="output", 
        help="Key name in the dataset for the output text (default: output)"
    )
    parser.add_argument(
        "--local_path", 
        type=str, 
        default="outputs", 
        help="Local path to save the trained model (default: outputs)"
    )
    parser.add_argument(
        "--hub_path", 
        type=str, 
        default="yu-ken0207/test", 
        help="Hugging Face Hub repository path (default: yu-ken0207/test)"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default="your_hf_token", 
        help="Hugging Face authentication token"
    )
    parser.add_argument(
        "--task_name", 
        type=str, 
        default="translation_task", 
        help="Task name for the saved model (default: translation_task)"
    )
    args = parser.parse_args()

    # 從參數讀取所有配置
    model_name = args.model_name
    dataset_name = args.dataset_name
    instruction = args.instruction
    input_key = args.input_key
    output_key = args.output_key
    local_path = args.local_path
    hub_path = args.hub_path
    token = args.token
    task_name = args.task_name

    # Load model
    llama = LlamaModel(model_name)
    lora_config = {
        "r": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_alpha": 16,
        "lora_dropout": 0,
        "use_gradient_checkpointing": "unsloth",
    }
    llama.apply_lora(lora_config)

    # Load dataset
    dataset_loader = DatasetLoader(dataset_name)
    dataset = dataset_loader.load_dataset()

    def formatting_func(examples):
        alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Input:
        {{}}
        
        ### Response:
        {{}}"""
        eos_token = llama.tokenizer.eos_token
        texts = [
            alpaca_prompt.format(inp, out) + eos_token
            for inp, out in zip(examples[input_key], examples[output_key])
        ]
        return {"text": texts}

    dataset = dataset_loader.format_dataset(formatting_func)

    # 印出資料集中的一筆範例
    print("Example from dataset:")
    print(dataset[0])  # 假設 dataset 支援索引訪問，列印第一筆資料

    # Train model
    train_model(llama.model, llama.tokenizer, dataset, llama.max_seq_length)

    # Save trained model
    llama.save_model(local_path, hub_path, token, task_name=task_name)

if __name__ == "__main__":
    main()
