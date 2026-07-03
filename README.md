# AxisMapper

A fine-tuned LLM that maps any natural language medical description to ICD-10-CM codes — structured output only, no explanation, no prose. Built to power insurance claim estimation pipelines where a wrong code means a denied claim.

---

## The Problem

Insurance reimbursement in the US runs on ICD-10-CM codes. A patient describes a cycling accident. A coder manually looks up the right codes. The codes map to a DRG. The DRG determines reimbursement.

That manual step is slow, expensive, and inconsistent. The gap is not a lookup problem — it's a language problem. Patients and doctors describe conditions in natural language. Insurance systems speak in codes.

AxisMapper closes that gap.

---

## What It Does

```
Input:  "I was cycling and a car ran over my leg. My tibia is fractured."
Output: S82.201A, V18.4XXA
```

Nothing else. No explanation. Just the codes.

The downstream pipeline — DRG grouping, reimbursement estimation — runs on those codes. The model's only job is to be the translation layer between language and the coding system.

---

## Links

- HuggingFace: [AmareshHebbar](https://huggingface.co/AmareshHebbar/icd10-coder-qwen25-7b-merged)
- W&B: [AmareshHebbar](https://api.wandb.ai/links/amareshhebbar-/qonl6s58)
- Data source: [CMS ICD-10-CM FY2026](https://www.cms.gov/medicare/coding-billing/icd-10-codes)
- MS-DRG v43.1: [CMS MS-DRG Definitions](https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/ms-drg-classifications-and-software)


## Stack

| Layer | Tool |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Adapter method | QLoRA → DoRA |
| Alignment | ORPO |
| Training framework | Unsloth + TRL |
| Training hardware | RTX A5000 24GB on RunPod (~$0.44/hr) |
| Experiment tracking | Weights & Biases |
| Model registry | HuggingFace Hub (private) |
| Serving | vLLM |
| Data source | CMS ICD-10-CM FY2026 (April update) + Llama 3.2 synthetic generation |

---

## Why Each Method Was Chosen

### QLoRA first, DoRA second

The base model (Qwen2.5-7B) already knows language and medical concepts from pretraining. Full fine-tuning would overwrite that — wasteful and expensive. LoRA freezes the base and trains two small matrices per layer. The insight behind LoRA is that weight updates during fine-tuning occupy a low-rank subspace anyway, so full-rank updates are unnecessary.

QLoRA goes further: it quantizes the frozen base to 4-bit (NF4 format), cutting VRAM from ~15GB to ~5GB. The adapters still train in bf16. This is what makes training on a single RTX A5000 possible.

DoRA improves on standard LoRA by decomposing pre-trained weights into magnitude and direction, then applying LoRA only to the directional component. In practice this closes the quality gap between LoRA and full fine-tuning. It costs nothing extra — one config flag. It becomes relevant after QLoRA's F1 plateaus.

The progression is deliberate: start cheap and fast, upgrade only when the data justifies it.

### ORPO for alignment

After SFT, the model knows which codes are correct but hasn't been explicitly penalized for outputting prose alongside them. A standard fine-tuned model might say:

> *"Based on the scenario, the relevant codes are S82.201A and V18.4XXA, which represent..."*

That's a correctly coded response that's useless in a pipeline. DPO could fix this, but DPO requires keeping a reference model copy in memory during training — nearly double the VRAM, and a separate training stage.

ORPO solves both problems. It combines SFT loss and preference alignment in a single training pass with no reference model. The rejected examples are trivially generated: the correct codes wrapped in explanatory prose. The model learns that outputting correct codes with explanation is worse than outputting codes alone. After ORPO, format compliance hits >99%.

### The chain of reasoning

```
QLoRA trains what to output (correct ICD-10 codes)
    ↓
DoRA improves how precisely it outputs them
    ↓
ORPO enforces the format (codes only, nothing else)
```

Each method solves a different failure mode. None of them are interchangeable.

---

## Training Pipeline

```
CMS ICD-10-CM FY2026 XML
    → parse_icd10_xml.py         → 72,000 billable codes (validator)
    → generate_synthetic.py      → Llama 3.2 local (Ollama) generates patient scenarios
    → validate_codes.py          → drop any hallucinated or non-billable codes
    → MinHash deduplication      → remove near-duplicate scenarios
    → build_sft_dataset.py       → train/val/test JSONL (90/5/5)
    → build_orpo_dataset.py      → chosen/rejected pairs

Training:
    sft_runpod.py   → QLoRA SFT, 3 epochs
    orpo_train.py   → ORPO alignment, 1 epoch
    merge_export.py → merge adapter → push merged model to HuggingFace
```

Synthetic data was generated entirely locally using Ollama + Llama 3.2 — no API costs. Every generated code was validated against the official CMS tabular list before entering training. A single invalid code in training teaches the model to hallucinate it in production, which means a denied claim.

---

## Hardware & Cost

| Stage | Hardware | Duration | Cost |
|---|---|---|---|
| Data pipeline | Fedora local | — | $0 |
| Smoke test | Fedora local | ~10 min | $0 |
| SFT (3 epochs) | RTX A5000 24GB (RunPod) | ~2.5 hr | ~$1.10 |
| ORPO alignment | RTX A5000 24GB (RunPod) | ~1 hr | ~$0.44 |
| Merge + eval | RTX A5000 24GB (RunPod) | ~30 min | ~$0.22 |
| **Total** | | | **~$1.76** |

The RTX A5000 was chosen specifically: 24GB VRAM handles QLoRA 7B at batch=4 comfortably, Community Cloud pricing keeps it under $0.50/hr, and the fp32 accumulation path on Ampere architecture is stable with bf16 training.

Training ran directly connected to HuggingFace Hub — adapter weights pushed after training, merged model pushed after export. No intermediate S3 buckets, no manual uploads.

---

## Evaluation

| Metric | Target | Description |
|---|---|---|
| Exact Match | >65% | All predicted codes match ground truth exactly |
| Code F1 | >84% | Harmonic mean of precision and recall on code sets |
| Format Compliance | >99% | Output contains no prose — codes only |
| Hallucination Rate | <1% | Codes not in official ICD-10-CM FY2026 codeset |

The hallucination metric is the most operationally important one. A high F1 with high hallucinations is worse than a lower F1 with zero hallucinations — an invalid code propagates into the DRG grouper and produces a wrong claim estimate.

---

## Serving

The merged model is served via vLLM — OpenAI-compatible API, handles batching and KV cache automatically. The ICD-10 → DRG mapping is a post-inference table join, not part of the model. The model's responsibility ends at producing valid codes.

```bash
vllm serve AmareshHebbar/axismapper-icd10-qwen25-7b-merged \
    --host 0.0.0.0 --port 8000 --dtype bfloat16
```

---

## Repository Structure

```
configs/        QLoRA, DoRA, ORPO hyperparameter configs
data/           raw XML, parsed JSON validator, JSONL datasets
eval/           evaluation scripts, metrics, inference router
scripts/        data pipeline (parse, generate, validate, build)
train/          SFT local, SFT RunPod, ORPO, merge/export, continual
```


*Part of the medical and insurance intelligence layer.*
```
```



<!-- docs pass: 2026-07-03T04:32:43Z -->

< docs pass retry: 2026-07-03T04:44:02Z -->
