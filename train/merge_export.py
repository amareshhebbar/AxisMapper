import os
from pathlib import Path
from dotenv import load_dotenv
from unsloth import FastLanguageModel

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN"):
    print("⚠️ WARNING: HF_TOKEN not found. Private models will fail to load.")


MODEL_PATH = "AmareshHebbar/icd10-coder-qwen25-7b-orpo-v1"
HF_REPO_MERGED = "AmareshHebbar/icd10-coder-merged-fp16"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

print(f"Loading base model and fusing adapter: {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=512,
    dtype=None,           
    load_in_4bit=False,  
    token=os.getenv("HF_TOKEN") 
)

print("\n[1/3] Saving 16-bit merged safetensors locally...")
model.save_pretrained_merged(
    str(OUTPUT_DIR / "icd10-merged-fp16"),
    tokenizer,
    save_method="merged_16bit"
)

print("\n[2/3] Converting to GGUF Q4_K_M format...")
model.save_pretrained_gguf(
    str(OUTPUT_DIR / "icd10-gguf"), 
    tokenizer,
    quantization_method="q4_k_m"
)

print(f"\n[3/3] Pushing merged model to Hugging Face Hub ({HF_REPO_MERGED})...")
model.push_to_hub_merged(
    HF_REPO_MERGED,
    tokenizer,
    save_method="merged_16bit",
    private=True,
    token=os.getenv("HF_TOKEN")
)

print("\n✅ Merge and export complete! Your model is ready for production.")