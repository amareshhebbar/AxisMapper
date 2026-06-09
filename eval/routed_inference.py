import torch
from unsloth import FastLanguageModel

# ── 1. Configuration & Adapter Map ───────────────────────────────────────────
BASE_MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 512

# Map chapters to their respective trained adapter paths or HF hub IDs
CHAPTER_ADAPTERS = {
    "respiratory": "your-org/icd10-respiratory-adapter",  # J-codes
    "injuries":    "your-org/icd10-injuries-adapter",     # S-T codes
    "cardio":      "your-org/icd10-cardiovascular-adapter", # I-codes
    "general":     "your-org/icd10-general-sft-adapter"   # Fallback SFT
}

# ── 2. The Simple Rule-Based Router ──────────────────────────────────────────
def route_clinical_scenario(text: str) -> str:
    """
    Analyzes keywords in clinical text to route to the correct adapter.
    In an enterprise system, this could be a lightweight classifier (e.g., BERT).
    """
    text_lower = text.lower()
    
    # Respiratory Keywords
    if any(w in text_lower for w in ["asthma", "pneumonia", "copd", "cough", "bronchitis", "lung"]):
        return "respiratory"
    
    # Injury / Trauma Keywords
    if any(w in text_lower for w in ["fracture", "laceration", "burn", "contusion", "sprain", "fall"]):
        return "injuries"
    
    # Cardiovascular Keywords
    if any(w in text_lower for w in ["infarction", "heart", "cardiac", "hypertension", "artery", "stroke"]):
        return "cardio"
    
    return "general"

# ── 3. Unified Inference Pipeline ────────────────────────────────────────────
def main():
    print("Loading Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print("Loading Chapter Adapters into memory...")
    # Load each adapter with a unique string handle
    for adapter_name, adapter_path in CHAPTER_ADAPTERS.items():
        try:
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            print(f"  Loaded adapter: [{adapter_name}]")
        except Exception as e:
            print(f"  Skipping [{adapter_name}] (Not found or unbuilt): {e}")

    # Example incoming clinical scenarios
    test_scenarios = [
        "Patient presents with acute shortness of breath and wheezing, suspected exacerbation of chronic obstructive asthma.",
        "A 45-year-old male was admitted following a ladder fall, resulting in a displaced fracture of the right tibia shaft.",
        "Routine checkup reveals persistent stage 2 essential hypertension with a family history of myocardial infarction."
    ]

    print("\n🚀 Running Routed Inference Engine:\n" + "="*50)
    
    for scenario in test_scenarios:
        # Step A: Dynamically determine the best adapter domain
        target_domain = route_clinical_scenario(scenario)
        
        # Step B: Instantly hot-swap the model's active weights
        if target_domain in model.peft_config:
            model.set_adapter(target_domain)
            active_adapter = target_domain
        else:
            model.set_adapter("general")
            active_adapter = "general (fallback)"
            
        print(f"\n📝 Scenario: {scenario}")
        print(f"🎯 Routed to Adapter: {active_adapter.upper()}")

        # Step C: Tokenize and generate output
        inputs = tokenizer(
            [f"Extract ICD-10 codes for: {scenario}\nCodes:"], 
            return_tensors="pt"
        ).to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=32, temperature=0.0)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🔮 Output: {decoded_output.split('Codes:')[-1].strip()}")
        print("-" * 50)

if __name__ == "__main__":
    main()