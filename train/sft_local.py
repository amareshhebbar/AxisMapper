import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 512

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN") or not os.getenv("WANDB_API_KEY"):
    print("⚠️ WARNING: HF_TOKEN or WANDB_API_KEY not found in environment.")

train_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
val_path = PROJECT_ROOT / "data" / "processed" / "val.jsonl"

for p in [train_path, val_path]:
    if not p.exists():
        raise FileNotFoundError(f"Required data file not found: {p}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,        
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

with open(train_path, "r") as f:
    train_list = [json.loads(line) for line in f if line.strip()]
with open(val_path, "r") as f:
    val_list = [json.loads(line) for line in f if line.strip()]

dataset = {
    "train": Dataset.from_list(train_list).select(range(min(100, len(train_list)))),
    "validation": Dataset.from_list(val_list).select(range(min(20, len(val_list)))),
}

print(f"Smoke-test  Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir="./outputs/smoke-test",
        max_steps=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),   
        bf16=torch.cuda.is_bf16_supported(),        
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=999999,
        report_to="none",
        max_seq_length=MAX_SEQ_LEN,
    ),
)

trainer.train()
print("\nSmoke test PASSED — training loop runs correctly\n")

FastLanguageModel.for_inference(model)
test_prompt = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": "You are a medical coding assistant. Respond ONLY with ICD-10-CM codes.",
        },
        {
            "role": "user",
            "content": "Cycling accident, car ran over my leg, tibia is fractured.",
        },
    ],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)
result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Test output: '{result}'")
print("Expected format: 'S82.201A, V18.4XXA'")