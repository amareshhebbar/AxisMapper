import torch
from unsloth import FastLanguageModel

BASE_MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 512

CHAPTER_ADAPTERS: dict[str, str] = {
    "respiratory": "your-org/icd10-respiratory-adapter",   
    "injuries":    "your-org/icd10-injuries-adapter",     
    "cardio":      "your-org/icd10-cardiovascular-adapter", 
    "general":     "your-org/icd10-general-sft-adapter",   
}

ROUTING_RULES: dict[str, list[str]] = {
    "respiratory": ["asthma", "pneumonia", "copd", "cough", "bronchitis", "lung"],
    "injuries":    ["fracture", "laceration", "burn", "contusion", "sprain", "fall"],
    "cardio":      ["infarction", "heart", "cardiac", "hypertension", "artery", "stroke"],
}


def route_clinical_scenario(text: str) -> str:
    text_lower = text.lower()
    for domain, keywords in ROUTING_RULES.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    return "general"


def main() -> None:
    print("Loading Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print("Loading Chapter Adapters into memory...")
    loaded_adapters: set[str] = set()
    for adapter_name, adapter_path in CHAPTER_ADAPTERS.items():
        try:
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            loaded_adapters.add(adapter_name)
            print(f"  ✅ Loaded adapter: [{adapter_name}]")
        except Exception as e:
            print(f"  ⚠️  Skipping [{adapter_name}] (not found or unbuilt): {e}")

    if "general" not in loaded_adapters:
        raise RuntimeError(
            "The 'general' fallback adapter failed to load. "
            "At least the fallback must be available before running inference."
        )

    test_scenarios = [
        "Patient presents with acute shortness of breath and wheezing, "
        "suspected exacerbation of chronic obstructive asthma.",
        "A 45-year-old male was admitted following a ladder fall, resulting "
        "in a displaced fracture of the right tibia shaft.",
        "Routine checkup reveals persistent stage 2 essential hypertension "
        "with a family history of myocardial infarction.",
    ]

    print("\n Running Routed Inference Engine:\n" + "=" * 50)

    for scenario in test_scenarios:
        target_domain = route_clinical_scenario(scenario)

        if target_domain in loaded_adapters:
            model.set_adapter(target_domain)
            active_adapter = target_domain
        else:
            model.set_adapter("general")
            active_adapter = "general (fallback)"

        print(f"\n Scenario: {scenario}")
        print(f" Routed to Adapter: {active_adapter.upper()}")

        prompt = f"Extract ICD-10 codes for: {scenario}\nCodes:"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f" Output: {decoded_output.split('Codes:')[-1].strip()}")
        print("-" * 50)


if __name__ == "__main__":
    main()