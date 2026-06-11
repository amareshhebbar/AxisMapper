import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

REJECTION_TEMPLATES = [
    "Based on the clinical scenario described, the relevant ICD-10-CM codes are: {codes}. "
    "These codes capture the primary diagnosis and external cause of injury.",
    "The appropriate ICD-10 codes for this case are {codes}. "
    "The first code represents the injury and the second indicates the mechanism.",
    "For this medical situation, I would assign: {codes}. "
    "This represents the diagnoses present in the scenario.",
    "Looking at the scenario, the following ICD-10-CM codes apply: {codes}.",
    "ICD-10 coding for this case: {codes} — covering the primary condition and associated factors.",
]


def make_orpo_example(scenario: str, codes: list[str]) -> dict:
    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]

    codes_str = ", ".join(sorted(codes))
    chosen = codes_str  # bare codes — the "good" response

    template = random.choice(REJECTION_TEMPLATES)
    rejected = template.format(codes=codes_str)

    return {
        "prompt": scenario,
        "chosen": chosen,
        "rejected": rejected,
    }


def build_orpo_splits(validated_jsonl: Path, out_dir: Path, seed: int = 42) -> None:
    if not validated_jsonl.exists():
        print(f"❌ Error: Cannot find input data file at {validated_jsonl}")
        return

    random.seed(seed) 

    with open(validated_jsonl, "r") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    random.shuffle(examples)

    n_train = int(len(examples) * 0.9)
    train_data = examples[:n_train]
    val_data = examples[n_train:]

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        path = out_dir / f"orpo_{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                orpo_ex = make_orpo_example(ex["scenario"], ex["codes"])
                f.write(json.dumps(orpo_ex) + "\n")
        print(f"✅ ORPO {split_name.upper():<5}: {len(split_data):>6} pairs → {path.name}")


if __name__ == "__main__":
    input_file = PROJECT_ROOT / "data" / "raw" / "raw_synthetic.jsonl"
    output_directory = PROJECT_ROOT / "data" / "dpo"

    print("Building ORPO Preference Alignment dataset...")
    build_orpo_splits(validated_jsonl=input_file, out_dir=output_directory)