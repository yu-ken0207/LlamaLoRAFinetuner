from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

def get_training_args(output_dir="outputs", max_steps=100):
    return TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
    )

def train_model(model, tokenizer, dataset, max_seq_length):
    training_args = get_training_args()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    return trainer.train()
