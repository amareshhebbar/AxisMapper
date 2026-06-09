import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import ORPOTrainer, ORPOConfig
from datasets import load_dataset
import wandb

PatchDPOTrainer()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN") or not os.getenv("WANDB_API_KEY"):
    print("⚠️ WARNING: HF_TOKEN or WANDB_API_KEY not found in environment.")

MODEL_PATH  = "AmareshHebbar/icd10-coder-qwen25-7b-v1"  
MAX_SEQ_LEN = 512
RUN_NAME    = "icd10-orpo-v1"
HF_REPO     = "AmareshHebbar/icd10-coder-qwen25-7b-orpo"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    token=os.getenv("HF_TOKEN")
)

train_path = PROJECT_ROOT / "data" / "dpo" / "orpo_train.jsonl"
val_path = PROJECT_ROOT / "data" / "dpo" / "orpo_val.jsonl"

dataset = load_dataset("json", data_files={
    "train": str(train_path),
    "validation": str(val_path)
})
print(f"ORPO Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

output_dir = PROJECT_ROOT / "outputs" / RUN_NAME

trainer = ORPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
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

if os.getenv("WANDB_API_KEY"):
    wandb.login(key=os.getenv("WANDB_API_KEY"))

trainer.train()
trainer.save_model(str(output_dir / "final"))

token = os.getenv("HF_TOKEN")
model.push_to_hub(f"{HF_REPO}-v1", private=True, token=token)
tokenizer.push_to_hub(f"{HF_REPO}-v1", private=True, token=token)
print(f"✅ ORPO alignment done. Adapter pushed to {HF_REPO}-v1")