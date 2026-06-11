import unsloth  

import os
import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from datasets import Dataset

MODEL_NAME  = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 512

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN") or not os.getenv("WANDB_API_KEY"):
    print("⚠️ WARNING: HF_TOKEN or WANDB_API_KEY not found in environment.")

train_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
val_path   = PROJECT_ROOT / "data" / "processed" / "val.jsonl"

for p in [train_path, val_path]:
    if not p.exists():
        raise FileNotFoundError(f"Required data file not found: {p}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0,       
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

with open(train_path) as f:
    train_list = [json.loads(l) for l in f if l.strip()]
with open(val_path) as f:
    val_list = [json.loads(l) for l in f if l.strip()]

train_list = train_list[:min(100, len(train_list))]
val_list   = val_list[:min(20,  len(val_list))]

hf_train_dataset = Dataset.from_list(train_list)
hf_val_dataset   = Dataset.from_list(val_list)

print("============ HF TRAIN DATASET=============")
print(hf_train_dataset[:4])
print("================================")

print(f"Smoke-test  Train: {len(train_list)} | Val: {len(val_list)}")


def formatting_func(example):
    formatted = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    print(formatted)
    return [formatted] if isinstance(formatted, str) else formatted

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=hf_train_dataset,
    eval_dataset=hf_val_dataset,   
    formatting_func=formatting_func,
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
        dataset_text_field="text",
    ),
)

trainer.train()
print("\nSmoke test PASSED — training loop runs correctly\n")

FastLanguageModel.for_inference(model)
test_prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a medical coding assistant. Respond ONLY with ICD-10-CM codes."},
        {"role": "user",   "content": "Cycling accident, car ran over my leg, tibia is fractured."},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

inputs  = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)
result  = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Test output: '{result}'")
print("Expected format: 'S82.201A, V18.4XXA'")