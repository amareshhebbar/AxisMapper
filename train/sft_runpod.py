
import json
import os
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN") or not os.getenv("WANDB_API_KEY"):
    print("⚠️ WARNING: HF_TOKEN or WANDB_API_KEY not found in environment.")

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 512
LORA_RANK = 16
USE_DORA = False
RUN_NAME = "icd10-qlora-v1" if not USE_DORA else "icd10-dora-v1"
HF_REPO = "AmareshHebbar/icd10-coder-qwen25-7b"

valid_codes_path = PROJECT_ROOT / "data" / "raw" / "icd10cm_codes_2026.json"
train_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
val_path = PROJECT_ROOT / "data" / "processed" / "val.jsonl"

for p in [valid_codes_path, train_path, val_path]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

with open(valid_codes_path, "r") as f:
    VALID_CODES = set(json.load(f).keys())

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    token=os.getenv("HF_TOKEN"),
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK * 2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_dora=USE_DORA,
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

print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")


class ICD10EvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        m = kwargs.get("model")
        tok = kwargs.get("tokenizer")
        if m is None or tok is None:
            return

        sample_size = min(50, len(dataset["validation"]))
        val_sample = dataset["validation"].select(range(sample_size))

        correct = total = hallucinations = format_ok = 0
        over_count = under_count = 0

        FastLanguageModel.for_inference(m)
        for ex in val_sample:
            messages = ex["messages"]
            expected = next(msg["content"] for msg in messages if msg["role"] == "assistant")
            exp_codes = {c.strip() for c in expected.split(",")}

            prompt = tok.apply_chat_template(
                [msg for msg in messages if msg["role"] != "assistant"],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = m.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True,  
                )
            pred_str = tok.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            pred_codes = {c.strip() for c in pred_str.split(",") if c.strip()}

            if pred_codes == exp_codes:
                correct += 1

            hallucinations += len(pred_codes - VALID_CODES)

            words = pred_str.replace(",", " ").split()
            has_prose = any(len(w) > 3 and w.isalpha() for w in words)
            if not has_prose:
                format_ok += 1

            if len(pred_codes) > len(exp_codes) + 2:
                over_count += 1
            if len(pred_codes) < len(exp_codes) - 1:
                under_count += 1

            total += 1

        m.train()
        metrics = {
            "eval/exact_match":       correct / total,
            "eval/format_compliance": format_ok / total,
            "eval/hallucinations":    hallucinations,
            "eval/over_coding":       over_count / total,
            "eval/under_coding":      under_count / total,
            "eval/step":              state.global_step,
        }
        wandb.log(metrics)
        print(
            f"\n[Step {state.global_step}] EM={correct/total:.1%} | "
            f"Format={format_ok/total:.1%} | Halluc={hallucinations} | "
            f"Over={over_count} | Under={under_count}\n"
        )


output_dir = PROJECT_ROOT / "outputs" / RUN_NAME

if os.getenv("WANDB_API_KEY"):
    wandb.login(key=os.getenv("WANDB_API_KEY"))

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    callbacks=[ICD10EvalCallback()],
    args=SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_seq_length=MAX_SEQ_LEN,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=10,
        report_to="wandb",
        run_name=RUN_NAME,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    ),
)

trainer.train()
trainer.save_model(str(output_dir / "final"))

token = os.getenv("HF_TOKEN")
model.push_to_hub(f"{HF_REPO}-v1", private=True, token=token)
tokenizer.push_to_hub(f"{HF_REPO}-v1", private=True, token=token)
print(f"✅ Done. Adapter pushed to {HF_REPO}-v1")