import unsloth  

import json
import os
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from trl import ORPOConfig, ORPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer

PatchDPOTrainer()

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN") or not os.getenv("WANDB_API_KEY"):
    print("⚠️ WARNING: HF_TOKEN or WANDB_API_KEY not found in environment.")

MODEL_PATH  = "AmareshHebbar/icd10-coder-qwen25-7b-v1"
MAX_SEQ_LEN = 512
RUN_NAME    = "icd10-orpo-v1"
HF_REPO     = "AmareshHebbar/icd10-coder-qwen25-7b-orpo"

train_path = PROJECT_ROOT / "data" / "dpo" / "orpo_train.jsonl"
val_path   = PROJECT_ROOT / "data" / "dpo" / "orpo_val.jsonl"

for p in [train_path, val_path]:
    if not p.exists():
        raise FileNotFoundError(
            f"ORPO data file not found: {p}\n"
            "Run scripts/build_orpo_dataset.py first."
        )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    token=os.getenv("HF_TOKEN"),
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

with open(train_path) as f:
    train_list = [json.loads(l) for l in f if l.strip()]
with open(val_path) as f:
    val_list = [json.loads(l) for l in f if l.strip()]

print(f"ORPO Train: {len(train_list)} | Val: {len(val_list)}")

output_dir = PROJECT_ROOT / "outputs" / RUN_NAME

if os.getenv("WANDB_API_KEY"):
    wandb.login(key=os.getenv("WANDB_API_KEY"))

trainer = ORPOTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_list,
    eval_dataset=val_list,
    args=ORPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.1,
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=512,
        max_prompt_length=256,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        report_to="wandb",
        run_name=RUN_NAME,
    ),
)

trainer.train()
trainer.save_model(str(output_dir / "final"))

token = os.getenv("HF_TOKEN")
model.push_to_hub(f"{HF_REPO}-v1", private=True, token=token)
tokenizer.push_to_hub(f"{HF_REPO}-v1", private=True, token=token)
print(f"✅ ORPO alignment done. Adapter pushed to {HF_REPO}-v1")