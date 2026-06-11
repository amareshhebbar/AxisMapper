import os
import json
import sys
import torch
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from unsloth import FastLanguageModel
from eval.metrics import compute_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN"):
    print("⚠️ WARNING: HF_TOKEN not found. Private models will fail to load.")

MODEL_PATH = "AmareshHebbar/icd10-coder-qwen25-7b-orpo-v1"
TEST_JSONL = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
CODES_JSON = PROJECT_ROOT / "data" / "raw" / "icd10cm_codes_2026.json"

for required in [TEST_JSONL, CODES_JSON]:
    if not required.exists():
        raise FileNotFoundError(f"Required file not found: {required}")

print(f"Loading Base Model and ORPO Adapter: {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=512,
    load_in_4bit=True,
    token=os.getenv("HF_TOKEN"),
)
FastLanguageModel.for_inference(model)

print("Loading test data and validation dictionaries...")
with open(CODES_JSON, "r") as f:
    valid_codes = set(json.load(f).keys())

with open(TEST_JSONL, "r") as f:
    test_data = [json.loads(l) for l in f if l.strip()]

results = []
print(f"Beginning inference on {len(test_data)} test scenarios...")

for ex in tqdm(test_data, desc="Evaluating", unit="scenario"):
    messages = ex["messages"]
    expected = next(m["content"] for m in messages if m["role"] == "assistant")
    exp_codes = {c.strip() for c in expected.split(",") if c.strip()}

    prompt = tokenizer.apply_chat_template(
        [m for m in messages if m["role"] != "assistant"],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
        )

    pred_str = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    pred_codes = {c.strip() for c in pred_str.split(",") if c.strip()}

    results.append(
        {
            "expected": exp_codes,
            "predicted": pred_codes,
            "hallucinated": pred_codes - valid_codes,
            "raw_output": pred_str,
        }
    )

metrics = compute_metrics(results)

print("\n" + "=" * 40)
print("🏆 FINAL EVALUATION RESULTS")
print("=" * 40)
print(json.dumps(metrics, indent=2))
print("=" * 40)