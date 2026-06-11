import unsloth  

import json
import os
import random
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN") or not os.getenv("WANDB_API_KEY"):
    print("⚠️ WARNING: HF_TOKEN or WANDB_API_KEY not found in environment.")


def create_continual_dataset(
    new_data_jsonl: Path,
    old_data_jsonl: Path,
    replay_ratio: float = 0.08,
    seed: int = 42,
) -> list[dict]:
    random.seed(seed)
    with open(new_data_jsonl) as f:
        new_examples = [json.loads(l) for l in f if l.strip()]
    with open(old_data_jsonl) as f:
        old_examples = [json.loads(l) for l in f if l.strip()]
    n_replay = int(len(new_examples) * replay_ratio)
    replay   = random.sample(old_examples, min(n_replay, len(old_examples)))
    combined = new_examples + replay
    random.shuffle(combined)
    return combined


def main() -> None:
    MODEL_PATH  = "AmareshHebbar/icd10-coder-qwen25-7b-orpo-v1"
    HF_REPO_V2  = "AmareshHebbar/icd10-coder-qwen25-7b-v2"
    MAX_SEQ_LEN = 512
    RUN_NAME    = "icd10-continual-v2"

    new_data_path = PROJECT_ROOT / "data" / "processed" / "icd11_update_train.jsonl"
    old_data_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"

    if not new_data_path.exists():
        print(f"⚠️  Future update data not found at {new_data_path}")
        print("Exiting early — add the data when ICD updates happen!")
        return
    if not old_data_path.exists():
        raise FileNotFoundError(f"Base training data not found: {old_data_path}")

    print(f"Loading previous best model ({MODEL_PATH})...")
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

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    print("Building Continual Learning Dataset (Replay)...")
    mixed_list = create_continual_dataset(new_data_path, old_data_path, replay_ratio=0.10)
    print(f"Training on {len(mixed_list)} examples...")
    def formatting_func(example):
        formatted = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        
        return [formatted] if isinstance(formatted, str) else formatted

    output_dir = PROJECT_ROOT / "outputs" / RUN_NAME

    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=mixed_list,
        formatting_func=formatting_func,
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_seq_length=MAX_SEQ_LEN,
            dataset_text_field="text",
            save_steps=500,
            logging_steps=10,
            report_to="wandb",
            run_name=RUN_NAME,
        ),
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))

    token = os.getenv("HF_TOKEN")
    model.push_to_hub(HF_REPO_V2, private=True, token=token)
    tokenizer.push_to_hub(HF_REPO_V2, private=True, token=token)
    print(f"✅ Continual update complete. Model pushed to {HF_REPO_V2}")


if __name__ == "__main__":
    main()