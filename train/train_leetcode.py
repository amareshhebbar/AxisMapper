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

HF_TOKEN       = os.getenv("HF_TOKEN", "").strip()
WANDB_API_KEY  = os.getenv("WANDB_API_KEY", "").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
RUNPOD_POD_ID  = os.getenv("RUNPOD_POD_ID", "").strip()

WORKSPACE = Path("/workspace/outputs")

TASKS = {
    "leetcode_python": {
        "dataset": "AmareshHebbar/leetcode-codegen-python",
        "model":   "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "output":  "AmareshHebbar/leetcode-python-qwen25-coder-7b",
        "max_seq": 2048, "epochs": 2, "batch": 4, "grad_accum": 8, "lr": 2e-4, "dora": False,
    },
    "leetcode_java": {
        "dataset": "AmareshHebbar/leetcode-codegen-java",
        "model":   "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "output":  "AmareshHebbar/leetcode-java-qwen25-coder-7b",
        "max_seq": 2048, "epochs": 2, "batch": 4, "grad_accum": 8, "lr": 2e-4, "dora": False,
    },
    "leetcode_cpp": {
        "dataset": "AmareshHebbar/leetcode-codegen-cpp",
        "model":   "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "output":  "AmareshHebbar/leetcode-cpp-qwen25-coder-7b",
        "max_seq": 2048, "epochs": 2, "batch": 4, "grad_accum": 8, "lr": 2e-4, "dora": False,
    },
    "leetcode_javascript": {
        "dataset": "AmareshHebbar/leetcode-codegen-javascript",
        "model":   "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "output":  "AmareshHebbar/leetcode-javascript-qwen25-coder-7b",
        "max_seq": 2048, "epochs": 2, "batch": 4, "grad_accum": 8, "lr": 2e-4, "dora": False,
    },
}


def fmt_time(s):
    return str(timedelta(seconds=int(s)))


def print_progress(i, total, task, t_start, results):
    done = i - 1
    elapsed = time.time() - t_start
    if done > 0:
        per_task = elapsed / done
        remaining = per_task * (total - done)
        eta = datetime.now() + timedelta(seconds=remaining)
        eta_str = eta.strftime("%I:%M %p")
    else:
        remaining = 0
        eta_str = "calculating..."
    bar = "█" * done + "░" * (total - done)
    print(f"\n{'='*60}")
    print(f"  PROGRESS  [{bar}]  {done}/{total}")
    print(f"  Elapsed:  {fmt_time(elapsed)}")
    print(f"  Remaining:{fmt_time(remaining)}")
    print(f"  ETA:      {eta_str}")
    print(f"  Current:  [{i}/{total}] {task}")
    if results:
        print(f"  Done:     " + ", ".join(r["task"] for r in results if r["pushed"]))
    print(f"{'='*60}\n")


def preflight(tasks_to_run, dry_run):
    errors = []
    print("\n[preflight]")
    if not torch.cuda.is_available():
        errors.append("No GPU found")
    else:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name}  {vram:.1f}GB")
        if vram < 20:
            errors.append(f"Only {vram:.1f}GB VRAM — need 24GB+")

    if not HF_TOKEN:
        errors.append("HF_TOKEN not set")
    else:
        from huggingface_hub import HfApi
        try:
            user = HfApi().whoami(token=HF_TOKEN)
            print(f"  HF user: {user['name']}")
        except Exception as e:
            errors.append(f"HF token invalid: {e}")

    if not WANDB_API_KEY:
        errors.append("WANDB_API_KEY not set")
    else:
        import wandb
        try:
            ok = wandb.login(key=WANDB_API_KEY, verify=True, relogin=True)
            if not ok:
                errors.append("WANDB_API_KEY invalid — W&B login failed")
            else:
                print("  W&B OK")
        except Exception as e:
            errors.append(f"W&B error: {e}")

    from huggingface_hub import HfApi
    api = HfApi()
    for task in tasks_to_run:
        ds_id = TASKS[task]["dataset"]
        try:
            api.dataset_info(ds_id, token=HF_TOKEN)
            print(f"  dataset OK: {ds_id}")
        except Exception as e:
            errors.append(f"Cannot access {ds_id}: {e}")

    try:
        free = shutil.disk_usage("/workspace").free / 1e9
        print(f"  disk free: {free:.0f}GB")
        if free < 10:
            errors.append(f"Low disk: {free:.1f}GB")
    except Exception:
        pass

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    {e}")
        sys.exit(1)

    print(f"\n  READY — {len(tasks_to_run)} tasks:")
    for t in tasks_to_run:
        cfg = TASKS[t]
        print(f"    {t:<25} {cfg['model'].split('/')[-1]:<40} batch={cfg['batch']} dora={cfg['dora']}")

    if dry_run:
        sys.exit(0)
    time.sleep(3)


def train_task(task_name, results, task_idx, total_tasks, t_start):
    import unsloth
    import wandb
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    cfg = TASKS[task_name]
    t_task = time.time()

    print_progress(task_idx, total_tasks, task_name, t_start, results)

    print(f"[1/5] Loading dataset: {cfg['dataset']}")
    ds = load_dataset(cfg["dataset"], token=HF_TOKEN)
    train_ds = ds["train"]
    val_ds = ds.get("validation", ds["train"].select(range(min(200, len(ds["train"])))))
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    print(f"[2/5] Loading model: {cfg['model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"], max_seq_length=cfg["max_seq"],
        dtype=torch.bfloat16, load_in_4bit=True, token=HF_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
        use_dora=cfg["dora"], random_state=42,
    )

    def fmt_fn(example):
        out = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return [out] if isinstance(out, str) else out

    wandb.init(project="leetcode-codegen", name=f"{task_name}-qlora", reinit="finish_previous")

    print(f"[3/5] Training — {cfg['epochs']} epochs  batch={cfg['batch']}x{cfg['grad_accum']}  dora={cfg['dora']}")
    out_dir = WORKSPACE / f"{task_name}-qlora"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds, formatting_func=fmt_fn,
        args=SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03,
            bf16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
            max_seq_length=cfg["max_seq"], dataset_text_field="text",
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            logging_steps=50, report_to="wandb",
            run_name=f"{task_name}-qlora",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
        ),
    )
    trainer.train()
    print(f"[4/5] Training complete — {fmt_time(time.time()-t_task)}")

    print(f"[5/5] Pushing to HF: {cfg['output']}")
    pushed = False
    try:
        model.push_to_hub(cfg["output"], token=HF_TOKEN, private=False)
        tokenizer.push_to_hub(cfg["output"], token=HF_TOKEN, private=False)
        print(f"  https://huggingface.co/{cfg['output']}")
        pushed = True
    except Exception as e:
        print(f"  Push failed: {e}")

    wandb.finish()

    try:
        if pushed and out_dir.exists():
            shutil.rmtree(out_dir)
    except Exception:
        pass

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_task
    results.append({
        "task": task_name, "output": cfg["output"],
        "url": f"https://huggingface.co/{cfg['output']}" if pushed else "FAILED",
        "time_hr": round(elapsed / 3600, 2), "pushed": pushed,
    })
    print(f"  {task_name} done in {fmt_time(elapsed)}\n")


def stop_pod():
    if not RUNPOD_API_KEY or not RUNPOD_POD_ID:
        print("[pod stop] skipped — RUNPOD_API_KEY/RUNPOD_POD_ID not set")
        return
    subprocess.run([
        "curl", "-s", "-X", "POST", "https://api.runpod.io/graphql",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {RUNPOD_API_KEY}",
        "--data", json.dumps({"query": f'mutation {{ podStop(input: {{podId: "{RUNPOD_POD_ID}"}}) {{ id }} }}'}),
    ], timeout=10)
    print("[pod stop] signal sent")


def print_summary(results, t_start):
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE — Total time: {fmt_time(elapsed)}")
    print(f"{'='*60}")
    for r in results:
        status = f"{r['url']}" if r["pushed"] else "FAILED"
        print(f"  {r['task']:<25} {r['time_hr']:>5.1f}h  {status}")
    failed = [r for r in results if not r["pushed"]]
    if failed:
        print(f"\n  {len(failed)} failed: {[r['task'] for r in failed]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_cleanup", action="store_true")
    args = parser.parse_args()

    tasks_to_run = list(TASKS.keys()) if args.task == "all" else [args.task]
    for t in tasks_to_run:
        if t not in TASKS:
            print(f"Unknown task: {t}. Options: {list(TASKS.keys())}")
            sys.exit(1)

    preflight(tasks_to_run, args.dry_run)

    t_start = time.time()
    results = []

    for i, task in enumerate(tasks_to_run, 1):
        try:
            train_task(task, results, i, len(tasks_to_run), t_start)
        except KeyboardInterrupt:
            print_summary(results, t_start)
            sys.exit(0)
        except Exception as e:
            print(f"{task} failed: {e}")
            results.append({"task": task, "output": TASKS[task]["output"],
                             "url": "EXCEPTION", "time_hr": 0, "pushed": False})
            continue

    if not args.no_cleanup:
        shutil.rmtree(WORKSPACE, ignore_errors=True)

    print_summary(results, t_start)
    stop_pod()


if __name__ == "__main__":
    main()