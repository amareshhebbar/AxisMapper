import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import unsloth  
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HF_TOKEN=os.getenv("HF_TOKEN", "").strip()
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "").strip()
HF_ORG = "AmareshHebbar"
WORKSPACE   =Path("/workspace/outputs")

TASKS = {
    "icd10_coder": {
        "dataset": "AmareshHebbar/icd10-coder-sft",
        "model":   "unsloth/Qwen2.5-7B-Instruct",
        "output":  "AmareshHebbar/icd10-coder-qwen25-7b",
        "max_seq": 512,  "epochs": 3, "batch": 4, "grad_accum": 4,  "lr": 2e-4,
    },
    "symptom_diagnoser": {
        "dataset": "AmareshHebbar/symptom-diagnoser-sft",
        "model":   "unsloth/Qwen2.5-7B-Instruct",
        "output":  "AmareshHebbar/symptom-diagnoser-qwen25-7b",
        "max_seq": 768,  "epochs": 2, "batch": 2, "grad_accum": 8,  "lr": 2e-4,
    },
    "clinical_summarizer": {
        "dataset": "AmareshHebbar/clinical-summarizer-sft",
        "model":   "unsloth/Qwen2.5-7B-Instruct",
        "output":  "AmareshHebbar/clinical-summarizer-qwen25-7b",
        "max_seq": 1024, "epochs": 3, "batch": 2, "grad_accum": 8,  "lr": 2e-4,
    },
    "snomed_mapper": {
        "dataset": "AmareshHebbar/snomed-mapper-sft",
        "model":   "unsloth/Qwen2.5-7B-Instruct",
        "output":  "AmareshHebbar/snomed-mapper-qwen25-7b",
        "max_seq": 512,  "epochs": 3, "batch": 4, "grad_accum": 4,  "lr": 2e-4,
    },
    "discharge_qa": {
        "dataset": "AmareshHebbar/discharge-qa-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/discharge-qa-qwen25-3b",
        "max_seq": 1024, "epochs": 3, "batch": 4, "grad_accum": 4,  "lr": 2e-4,
    },
    "radiology_coder": {
        "dataset": "AmareshHebbar/radiology-coder-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/radiology-coder-qwen25-3b",
        "max_seq": 1024, "epochs": 3, "batch": 4, "grad_accum": 4,  "lr": 2e-4,
    },
    "medical_ner": {
        "dataset": "AmareshHebbar/medical-ner-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/medical-ner-qwen25-3b",
        "max_seq": 512,  "epochs": 3, "batch": 4, "grad_accum": 4,  "lr": 2e-4,
    },
    "hindi_medical": {
        "dataset": "AmareshHebbar/hindi-medical-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/hindi-medical-qwen25-3b",
        "max_seq": 768,  "epochs": 3, "batch": 4, "grad_accum": 4,  "lr": 2e-4,
    },
    "cpt_coder": {
        "dataset": "AmareshHebbar/cpt-coder-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/cpt-coder-qwen25-3b",
        "max_seq": 512,  "epochs": 3, "batch": 4, "grad_accum": 4,  "lr": 2e-4,
    },
    "medical_billing": {
        "dataset": "AmareshHebbar/medical-billing-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/medical-billing-qwen25-3b",
        "max_seq": 512,  "epochs": 3, "batch": 4, "grad_accum": 4,  "lr": 2e-4,
    },
    "pmjay_classifier": {
        "dataset": "AmareshHebbar/pmjay-classifier-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/pmjay-classifier-qwen25-3b",
        "max_seq": 512,  "epochs": 3, "batch": 4, "grad_accum": 4,  "lr": 2e-4,
    },
    "pharmacy_ner": {
        "dataset": "AmareshHebbar/pharmacy-ner-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/pharmacy-ner-qwen25-1b",
        "max_seq": 512,  "epochs": 3, "batch": 8, "grad_accum": 4,  "lr": 2e-4,
    },
    "ayurveda_icd": {
        "dataset": "AmareshHebbar/ayurveda-icd-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/ayurveda-icd-qwen25-1b",
        "max_seq": 512,  "epochs": 3, "batch": 8, "grad_accum": 4,  "lr": 2e-4,
    },
    "insurance_classifier": {
        "dataset": "AmareshHebbar/insurance-classifier-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/insurance-classifier-qwen25-1b",
        "max_seq": 512,  "epochs": 3, "batch": 8, "grad_accum": 4,  "lr": 2e-4,
    },
    "icd10_to_drg": {
        "dataset": "AmareshHebbar/icd10-to-drg-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/icd10-to-drg-qwen25-1b",
        "max_seq": 512,  "epochs": 3, "batch": 8, "grad_accum": 4,  "lr": 2e-4,
    },
    "loinc_coder": {
        "dataset": "AmareshHebbar/loinc-coder-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/loinc-coder-qwen25-1b",
        "max_seq": 512,  "epochs": 3, "batch": 8, "grad_accum": 4,  "lr": 2e-4,
    },
}



def preflight(tasks_to_run: list, skip_wandb: bool, dry_run: bool):
    errors = []
    warnings = []

    print("\n" + "="*60)
    print("  PRE-FLIGHT CHECKS")
    print("="*60)

    print("\n[1/5] GPU check...")
    if not torch.cuda.is_available():
        errors.append("No GPU found — torch.cuda.is_available() = False")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb=torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ {gpu_name}  —  {vram_gb:.1f} GB VRAM")
        if vram_gb < 16:
            errors.append(f"GPU has only {vram_gb:.1f}GB VRAM — need ≥16GB for QLoRA 7B")

    print("\n[2/5] HuggingFace token check...")
    if not HF_TOKEN:
        errors.append("HF_TOKEN not set in .env — model push will fail")
    else:
        try:
            
            api = HfApi()
            user = api.whoami(token=HF_TOKEN)
            print(f"  ✓ Logged in as: {user['name']}")
            if user["name"] != HF_ORG:
                warnings.append(f"HF user is '{user['name']}' but HF_ORG is '{HF_ORG}' — make sure token has write access to {HF_ORG}")
        except Exception as e:
            errors.append(f"HF token invalid: {e}")

    print("\n[3/5] Weights & Biases check...")
    if skip_wandb:
        print("  ~ Skipped (--skip_wandb flag set)")
    elif not WANDB_API_KEY:
        warnings.append("WANDB_API_KEY not set — training will run without W&B logging")
        print("  ~ Not set — continuing without W&B")
    else:
        try:
            import wandb
            result = wandb.login(key=WANDB_API_KEY, verify=True, relogin=True)
            if result:
                print(f"  ✓ W&B login successful")
            else:
                errors.append("W&B login failed — check WANDB_API_KEY")
        except Exception as e:
            errors.append(f"W&B error: {e}")

    print("\n[4/5] Dataset availability check...")
    try:
        
        api = HfApi()
        for task in tasks_to_run[:3]: 
            ds_id = TASKS[task]["dataset"]
            try:
                info = api.dataset_info(ds_id, token=HF_TOKEN)
                print(f"  ✓ {ds_id} — {info.card_data.get('size_categories', ['?'])[0] if info.card_data else '?'}")
            except Exception as e:
                errors.append(f"Cannot access dataset {ds_id}: {e}")
    except Exception as e:
        warnings.append(f"Dataset check skipped: {e}")

    print("\n[5/5] Disk space check...")
    try:
        stat = shutil.disk_usage("/workspace")
        free_gb = stat.free / 1e9
        print(f"  /workspace: {free_gb:.1f} GB free")
        if free_gb < 10:
            errors.append(f"Low disk space: {free_gb:.1f}GB free on /workspace — need ≥10GB")
        else:
            print(f"  ✓ Sufficient disk space")
    except Exception:
        warnings.append("Could not check disk space on /workspace")

    print("\n" + "-"*60)
    if warnings:
        print("  WARNINGS:")
        for w in warnings:
            print(f"    ⚠  {w}")

    if errors:
        print("\n  ERRORS — cannot continue:")
        for e in errors:
            print(f"    ✗  {e}")
        print("\n  Fix the above and re-run. Exiting.\n")
        sys.exit(1)

    print(f"\n  READY — will train {len(tasks_to_run)} task(s):")
    for task in tasks_to_run:
        cfg = TASKS[task]
        print(f"    {task:<25} {cfg['model'].split('/')[-1]:<30} → {cfg['output']}")

    if dry_run:
        print("\n  --dry_run set — exiting without training.\n")
        sys.exit(0)

    print(f"\n  Starting in 5 seconds... (Ctrl+C to abort)")
    time.sleep(5)
    print("="*60 + "\n")



def train_task(task_name: str, skip_wandb: bool, results: list):
    cfg = TASKS[task_name]
    model_name = cfg["model"]
    dataset_id = cfg["dataset"]
    hf_output=cfg["output"]
    run_name =f"{task_name}-qlora"
    output_dir = WORKSPACE / run_name

    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Task: {task_name}")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_id}")
    print(f"  Output: {hf_output}")
    print(f"{'='*60}")

    print(f"\n[1/5] Loading dataset...")
    ds=load_dataset(dataset_id, token=HF_TOKEN)
    train_ds = ds["train"]
    val_ds =ds.get("validation", ds.get("val",
               ds["train"].select(range(min(500, len(ds["train"]))))))
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    print(f"\n[2/5] Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg["max_seq"],
        dtype=torch.bfloat16,
        load_in_4bit=True,
        token=HF_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_dora=False,
        random_state=42,
    )

    def formatting_func(example):
        formatted = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return [formatted] if isinstance(formatted, str) else formatted

    use_wandb = WANDB_API_KEY and not skip_wandb
    if use_wandb:
        import wandb
        wandb.init(project="axiomapper", name=run_name, reinit=True)

    print(f"\n[3/5] Training — {cfg['epochs']} epochs, batch={cfg['batch']}×{cfg['grad_accum']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["lr"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_seq_length=cfg["max_seq"],
            dataset_text_field="text",
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=1,      
            logging_steps=10,
            report_to="wandb" if use_wandb else "none",
            run_name=run_name,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    trainer.train()
    print(f"\n[4/5] Training complete")

    print(f"\n[5/5] Pushing to HuggingFace: {hf_output}")
    push_success = False
    try:
        model.push_to_hub(hf_output, token=HF_TOKEN, private=False)
        tokenizer.push_to_hub(hf_output, token=HF_TOKEN, private=False)
        hf_url = f"https://huggingface.co/{hf_output}"
        print(f"  ✓ {hf_url}")
        push_success = True
    except Exception as e:
        print(f"  ✗ HF push failed: {e}")
        print(f"  Adapter saved locally at: {output_dir}")

    t_elapsed = time.time() - t_start
    print(f"\n  Cleaning up...")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    try:
        if push_success and output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"  ✓ Deleted local checkpoints: {output_dir}")
        else:
            print(f"  ~ Keeping local checkpoints (push failed): {output_dir}")
    except Exception as e:
        print(f"  ~ Could not delete {output_dir}: {e}")

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    vram_free = torch.cuda.memory_reserved(0) / 1e9
    print(f"  ✓ GPU memory cleared (reserved: {vram_free:.1f}GB)")

    results.append({
        "task":    task_name,
        "output":  hf_output,
        "url":     f"https://huggingface.co/{hf_output}" if push_success else "FAILED",
        "time_hr": round(t_elapsed / 3600, 2),
        "pushed":  push_success,
    })

    print(f"\n  ✓ {task_name} done in {t_elapsed/3600:.1f} hr")



def final_cleanup():
    print(f"\n[CLEANUP] Deleting /workspace/outputs/...")
    try:
        if WORKSPACE.exists():
            shutil.rmtree(WORKSPACE)
            print(f"  ✓ Deleted {WORKSPACE}")
        else:
            print(f"  ~ {WORKSPACE} already clean")
    except Exception as e:
        print(f"  ~ Could not delete {WORKSPACE}: {e}")


def print_summary(results: list, total_start: float):
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {total_time/3600:.1f} hr")
    print(f"{'='*60}")
    print(f"\n  {'Task':<25} {'Time':>6}  {'Status':<10}  URL")
    print(f"  {'-'*58}")
    for r in results:
        status = "✓ pushed" if r["pushed"] else "✗ failed"
        print(f"  {r['task']:<25} {r['time_hr']:>5.1f}h  {status:<10}  {r['url']}")

    failed = [r for r in results if not r["pushed"]]
    if failed:
        print(f"\n  ⚠  {len(failed)} task(s) failed to push:")
        for r in failed:
            print(f"     {r['task']} — adapter may be in /workspace/outputs/{r['task']}-qlora/")

    print(f"\n  All models: https://huggingface.co/{HF_ORG}")
    print()



def main():
    parser = argparse.ArgumentParser(description="AxisMapper universal trainer")
    parser.add_argument("--task",       required=True,
                        help=f"Task name or 'all'. Options: {', '.join(TASKS.keys())}")
    parser.add_argument("--skip_wandb", action="store_true",
                        help="Skip W&B logging entirely")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Run pre-flight checks only, do not train")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Keep /workspace/outputs after training (for debugging)")
    args = parser.parse_args()

    tasks_to_run = list(TASKS.keys()) if args.task == "all" else [args.task]

    for t in tasks_to_run:
        if t not in TASKS:
            print(f"[error] Unknown task: {t}")
            print(f"Available: {', '.join(TASKS.keys())}")
            sys.exit(1)

    preflight(tasks_to_run, args.skip_wandb, args.dry_run)

    total_start = time.time()
    results = []

    for i, task in enumerate(tasks_to_run, 1):
        print(f"\n[{i}/{len(tasks_to_run)}] Starting: {task}")
        try:
            train_task(task, args.skip_wandb, results)
        except KeyboardInterrupt:
            print(f"\n\n  Interrupted at task {task}. Results so far:")
            print_summary(results, total_start)
            sys.exit(0)
        except Exception as e:
            print(f"\n  ✗ Task {task} failed with exception: {e}")
            results.append({
                "task": task, "output": TASKS[task]["output"],
                "url": "EXCEPTION", "time_hr": 0, "pushed": False,
            })
            print(f"  Continuing to next task...\n")
            continue

    if not args.no_cleanup:
        final_cleanup()

    print_summary(results, total_start)


if __name__ == "__main__":
    main()