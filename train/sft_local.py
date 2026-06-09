from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import wandb
from dotenv import load_dotenv
import json
from pathlib import Path
import os
from datasets import Dataset

# MODEL_NAME   = "unsloth/Qwen2.5-7B-Instruct"
MODEL_NAME   = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN  = 512


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN") or not os.getenv("WANDB_API_KEY"):
    print("⚠️ WARNING: HF_TOKEN or WANDB_API_KEY not found in environment.")
    

valid_codes_path = PROJECT_ROOT / "data" / "raw" / "icd10cm_codes_2026.json" 

if not valid_codes_path.exists():
    raise FileNotFoundError(f"Missing validator file: {valid_codes_path}")


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

train_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
val_path = PROJECT_ROOT / "data" / "processed" / "val.jsonl"

with open(train_path, "r") as f:
    train_list = [json.loads(line) for line in f if line.strip()]
with open(val_path, "r") as f:
    val_list = [json.loads(line) for line in f if line.strip()]

dataset = {
    "train": Dataset.from_list(train_list).select(range(100)),
    "validation": Dataset.from_list(val_list).select(range(20))
}

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
        fp16=True,                     
        bf16=False,
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
test_prompt = tokenizer.apply_chat_template([
    {"role": "system",  "content": "You are a medical coding assistant. Respond ONLY with ICD-10-CM codes."},
    {"role": "user",    "content": "Cycling accident, car ran over my leg, tibia is fractured."}
], tokenize=False, add_generation_prompt=True)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=False)
result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Test output: '{result}'")
print("Expected format: 'S82.201A, V18.4XXA'")