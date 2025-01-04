from unsloth import FastLanguageModel
import torch
import datetime

class LlamaModel:
    def __init__(self, model_name, max_seq_length=2048, dtype=None, load_in_4bit=True):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        return FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )

    def apply_lora(self, config):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=config.get("r", 16),
            target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0),
            bias=config.get("bias", "none"),
            use_gradient_checkpointing=config.get("use_gradient_checkpointing", "unsloth"),
            random_state=config.get("random_state", 3407),
            use_rslora=config.get("use_rslora", False),
            loftq_config=config.get("loftq_config", None),
        )

    def save_model(self, local_path, hub_path, token, task_name=None):
        # 生成保存名稱，加入時間戳、任務名稱或步數
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_name}"
        if task_name:
            model_name += f"_{task_name}"
        model_name += f"_{timestamp}"

        # 本地保存的完整路徑
        save_path = f"{local_path}/{model_name}"

        # 保存模型和分詞器
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # 推送到 Hugging Face Hub
        self.model.push_to_hub(hub_path, token=token)
        self.tokenizer.push_to_hub(hub_path, token=token)
