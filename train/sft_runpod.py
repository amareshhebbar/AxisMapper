import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HF_TOKEN      = os.getenv("HF_TOKEN", "").strip()
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
RUNPOD_POD_ID  = os.getenv("RUNPOD_POD_ID", "").strip()
HF_ORG        = "AmareshHebbar"
WORKSPACE     = Path("/workspace/outputs")

TASKS = {

    "symptom_diagnoser": {
        "dataset": "AmareshHebbar/symptom-diagnoser-sft",
        "model":   "unsloth/Qwen2.5-7B-Instruct",
        "output":  "AmareshHebbar/symptom-diagnoser-qwen25-7b",
        "max_seq": 512, "epochs": 2, "batch": 8, "grad_accum": 4, "lr": 2e-4,
    },
    "clinical_summarizer": {
        "dataset": "AmareshHebbar/clinical-summarizer-sft",
        "model":   "unsloth/Qwen2.5-7B-Instruct",
        "output":  "AmareshHebbar/clinical-summarizer-qwen25-7b",
        "max_seq": 512, "epochs": 1, "batch": 8, "grad_accum": 4, "lr": 2e-4,
    },
    "snomed_mapper": {
        "dataset": "AmareshHebbar/snomed-mapper-sft",
        "model":   "unsloth/Qwen2.5-7B-Instruct",
        "output":  "AmareshHebbar/snomed-mapper-qwen25-7b",
        "max_seq": 512, "epochs": 1, "batch": 8, "grad_accum": 4, "lr": 2e-4,
    },
    "discharge_qa": {
        "dataset": "AmareshHebbar/discharge-qa-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/discharge-qa-qwen25-3b",
        "max_seq": 512, "epochs": 3, "batch": 12, "grad_accum": 4, "lr": 2e-4,
    },
    "radiology_coder": {
        "dataset": "AmareshHebbar/radiology-coder-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/radiology-coder-qwen25-3b",
        "max_seq": 512, "epochs": 3, "batch": 12, "grad_accum": 4, "lr": 2e-4,
    },
    "medical_ner": {
        "dataset": "AmareshHebbar/medical-ner-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/medical-ner-qwen25-3b",
        "max_seq": 512, "epochs": 3, "batch": 12, "grad_accum": 4, "lr": 2e-4,
    },
    "hindi_medical": {
        "dataset": "AmareshHebbar/hindi-medical-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/hindi-medical-qwen25-3b",
        "max_seq": 512, "epochs": 3, "batch": 12, "grad_accum": 4, "lr": 2e-4,
    },
    "cpt_coder": {
        "dataset": "AmareshHebbar/cpt-coder-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/cpt-coder-qwen25-3b",
        "max_seq": 512, "epochs": 3, "batch": 12, "grad_accum": 4, "lr": 2e-4,
    },
    "medical_billing": {
        "dataset": "AmareshHebbar/medical-billing-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/medical-billing-qwen25-3b",
        "max_seq": 512, "epochs": 3, "batch": 12, "grad_accum": 4, "lr": 2e-4,
    },
    "pmjay_classifier": {
        "dataset": "AmareshHebbar/pmjay-classifier-sft",
        "model":   "unsloth/Qwen2.5-3B-Instruct",
        "output":  "AmareshHebbar/pmjay-classifier-qwen25-3b",
        "max_seq": 512, "epochs": 3, "batch": 12, "grad_accum": 4, "lr": 2e-4,
    },
    "pharmacy_ner": {
        "dataset": "AmareshHebbar/pharmacy-ner-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/pharmacy-ner-qwen25-1b",
        "max_seq": 512, "epochs": 3, "batch": 16, "grad_accum": 4, "lr": 2e-4,
    },
    "ayurveda_icd": {
        "dataset": "AmareshHebbar/ayurveda-icd-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/ayurveda-icd-qwen25-1b",
        "max_seq": 512, "epochs": 3, "batch": 16, "grad_accum": 4, "lr": 2e-4,
    },
    "insurance_classifier": {
        "dataset": "AmareshHebbar/insurance-classifier-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/insurance-classifier-qwen25-1b",
        "max_seq": 512, "epochs": 3, "batch": 16, "grad_accum": 4, "lr": 2e-4,
    },
    "icd10_to_drg": {
        "dataset": "AmareshHebbar/icd10-to-drg-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/icd10-to-drg-qwen25-1b",
        "max_seq": 512, "epochs": 3, "batch": 16, "grad_accum": 4, "lr": 2e-4,
    },
    "loinc_coder": {
        "dataset": "AmareshHebbar/loinc-coder-sft",
        "model":   "unsloth/Qwen2.5-1.5B-Instruct",
        "output":  "AmareshHebbar/loinc-coder-qwen25-1b",
        "max_seq": 512, "epochs": 3, "batch": 16, "grad_accum": 4, "lr": 2e-4,
    },
        "icd10_coder": {
        "dataset": "AmareshHebbar/icd10-coder-sft",
        "model":   "unsloth/Qwen2.5-7B-Instruct",
        "output":  "AmareshHebbar/icd10-coder-qwen25-7b",
        "max_seq": 512, "epochs": 1, "batch": 8, "grad_accum": 4, "lr": 2e-4,
    },
}

def fmt_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def print_progress(i, total, task, t_start, results):
    done    = i - 1
    elapsed = time.time() - t_start
    if done > 0:
        per_task = elapsed / done
        remaining = per_task * (total - done)
        eta = datetime.now() + timedelta(seconds=remaining)
        eta_str = eta.strftime("%I:%M %p")
    else:
        remaining = 0
        eta_str = "calculating..."

    bar_filled = "█" * done + "░" * (total - done)
    print(f"\n{'='*60}")
    print(f"  PROGRESS  [{bar_filled}]  {done}/{total}")
    print(f"  Elapsed:  {fmt_time(elapsed)}")
    print(f"  Remaining:{fmt_time(remaining)}")
    print(f"  ETA:      {eta_str}")
    print(f"  Current:  [{i}/{total}] {task}")
    if results:
        print(f"  Done:     " + ", ".join(r["task"] for r in results if r["pushed"]))
    print(f"{'='*60}\n")


def preflight(tasks_to_run, skip_wandb, dry_run):
    errors = []
    print("\n" + "="*60)
    print("  PRE-FLIGHT CHECKS")
    print("="*60)

    print("\n[1/5] GPU...")
    if not torch.cuda.is_available():
        errors.append("No GPU found")
    else:
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ {name}  {vram:.1f}GB VRAM")
        if vram < 16:
            errors.append(f"Only {vram:.1f}GB VRAM — need 24GB+")

    print("\n[2/5] HuggingFace token...")
    if not HF_TOKEN:
        errors.append("HF_TOKEN not set")
    else:
        try:
            from huggingface_hub import HfApi
            user = HfApi().whoami(token=HF_TOKEN)
            print(f"  ✓ {user['name']}")
        except Exception as e:
            errors.append(f"HF token invalid: {e}")

    print("\n[3/5] Weights & Biases...")
    if skip_wandb:
        print("  ~ Skipped")
    elif not WANDB_API_KEY:
        print("  ~ No key — running without W&B")
    else:
        try:
            import wandb
            ok = wandb.login(key=WANDB_API_KEY, verify=True, relogin=True)
            print(f"  {'✓ OK' if ok else '✗ FAILED'}")
            if not ok:
                errors.append("W&B login failed")
        except Exception as e:
            errors.append(f"W&B error: {e}")

    print("\n[4/5] Datasets...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for task in tasks_to_run[:3]:
            ds_id = TASKS[task]["dataset"]
            try:
                api.dataset_info(ds_id, token=HF_TOKEN)
                print(f"  ✓ {ds_id}")
            except Exception as e:
                errors.append(f"Cannot access {ds_id}: {e}")
    except Exception as e:
        print(f"  ~ Dataset check skipped: {e}")

    print("\n[5/5] Disk space...")
    try:
        free = shutil.disk_usage("/workspace").free / 1e9
        print(f"  ✓ {free:.0f}GB free")
        if free < 10:
            errors.append(f"Low disk: {free:.1f}GB")
    except Exception:
        print("  ~ Cannot check disk")

    print(f"\n{'='*60}")
    if errors:
        print("  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        print("\n  Fix and re-run.\n")
        sys.exit(1)

    print(f"\n  READY — {len(tasks_to_run)} tasks:")
    for t in tasks_to_run:
        cfg = TASKS[t]
        size = cfg["model"].split("/")[-1]
        print(f"    {t:<25} {size:<35} batch={cfg['batch']}")

    if dry_run:
        print("\n  --dry_run — exiting.\n")
        sys.exit(0)

    print(f"\n  Starting in 3 seconds...")
    time.sleep(3)
    print("="*60 + "\n")


def train_task(task_name, skip_wandb, results, task_idx, total_tasks, t_start):
    import unsloth
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    cfg       = TASKS[task_name]
    t_task    = time.time()

    print_progress(task_idx, total_tasks, task_name, t_start, results)

    print(f"[1/5] Loading dataset: {cfg['dataset']}")
    ds       = load_dataset(cfg["dataset"], token=HF_TOKEN)
    train_ds = ds["train"]
    val_ds   = ds.get("validation", ds["train"].select(range(min(200, len(ds["train"])))))
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    print(f"[2/5] Loading model: {cfg['model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"],
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
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
        use_dora=False, random_state=42,
    )

    def fmt_fn(example):
        out = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return [out] if isinstance(out, str) else out

    use_wandb = bool(WANDB_API_KEY) and not skip_wandb
    if use_wandb:
        import wandb
        wandb.init(project="axiomapper", name=f"{task_name}-qlora", reinit="finish_previous")

    print(f"[3/5] Training — {cfg['epochs']} epochs  batch={cfg['batch']}x{cfg['grad_accum']}  max_seq={cfg['max_seq']}")
    out_dir = WORKSPACE / f"{task_name}-qlora"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=fmt_fn,
        args=SFTConfig(
            output_dir=str(out_dir),
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
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=50,
            report_to="wandb" if use_wandb else "none",
            run_name=f"{task_name}-qlora",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    trainer.train()
    print(f"[4/5] Training complete — {fmt_time(time.time()-t_task)}")

    print(f"[5/5] Pushing to HF: {cfg['output']}")
    pushed = False
    try:
        model.push_to_hub(cfg["output"], token=HF_TOKEN, private=False)
        tokenizer.push_to_hub(cfg["output"], token=HF_TOKEN, private=False)
        print(f"  ✓ https://huggingface.co/{cfg['output']}")
        pushed = True
    except Exception as e:
        print(f"  ✗ Push failed: {e}")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    try:
        if pushed and out_dir.exists():
            shutil.rmtree(out_dir)
            print(f"  ✓ Checkpoints deleted")
    except Exception:
        pass

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_task
    results.append({
        "task": task_name, "output": cfg["output"],
        "url": f"https://huggingface.co/{cfg['output']}" if pushed else "FAILED",
        "time_hr": round(elapsed/3600, 2), "pushed": pushed,
    })
    print(f"  ✓ {task_name} done in {fmt_time(elapsed)}\n")


def stop_pod():
    if not RUNPOD_API_KEY or not RUNPOD_POD_ID:
        print("  [pod stop] RUNPOD_API_KEY or RUNPOD_POD_ID not set — skipping auto-stop")
        return
    print(f"  [pod stop] Stopping pod {RUNPOD_POD_ID}...")
    try:
        subprocess.run([
            "curl", "-s", "-X", "POST", "https://api.runpod.io/graphql",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {RUNPOD_API_KEY}",
            "--data", json.dumps({"query": f'mutation {{ podStop(input: {{podId: "{RUNPOD_POD_ID}"}}) {{ id }} }}'})
        ], timeout=10)
        print("  ✓ Pod stop signal sent")
    except Exception as e:
        print(f"  ~ Pod stop failed: {e}")


def print_summary(results, t_start):
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE — Total time: {fmt_time(elapsed)}")
    print(f"{'='*60}")
    print(f"\n  {'Task':<25} {'Time':>6}  {'Status'}")
    print(f"  {'-'*45}")
    for r in results:
        status = f"✓ {r['url']}" if r["pushed"] else "✗ FAILED"
        print(f"  {r['task']:<25} {r['time_hr']:>5.1f}h  {status}")
    failed = [r for r in results if not r["pushed"]]
    if failed:
        print(f"\n  {len(failed)} failed: {[r['task'] for r in failed]}")
    print(f"\n  Profile: https://huggingface.co/{HF_ORG}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       required=True)
    parser.add_argument("--skip_wandb", action="store_true")
    parser.add_argument("--dry_run",    action="store_true")
    parser.add_argument("--no_cleanup", action="store_true")
    args = parser.parse_args()

    tasks_to_run = list(TASKS.keys()) if args.task == "all" else [args.task]
    for t in tasks_to_run:
        if t not in TASKS:
            print(f"[error] Unknown task: {t}. Options: {list(TASKS.keys())}")
            sys.exit(1)

    preflight(tasks_to_run, args.skip_wandb, args.dry_run)

    t_start = time.time()
    results = []

    for i, task in enumerate(tasks_to_run, 1):
        try:
            train_task(task, args.skip_wandb, results, i, len(tasks_to_run), t_start)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            print_summary(results, t_start)
            sys.exit(0)
        except Exception as e:
            print(f"\n  ✗ {task} failed: {e}")
            results.append({"task": task, "output": TASKS[task]["output"],
                            "url": "EXCEPTION", "time_hr": 0, "pushed": False})
            continue

    if not args.no_cleanup:
        try:
            if WORKSPACE.exists():
                shutil.rmtree(WORKSPACE)
                print(f"  ✓ /workspace/outputs deleted")
        except Exception:
            pass

    print_summary(results, t_start)
    stop_pod()


if __name__ == "__main__":
    main()