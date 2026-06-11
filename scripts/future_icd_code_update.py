import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def create_continual_dataset(
    new_data_jsonl: Path,   
    old_data_jsonl: Path,
    replay_ratio: float = 0.08,
    seed: int = 42,        
) -> list[dict]:
    """Mix new data with a small replay sample of old data to prevent forgetting."""
    random.seed(seed)

    with open(new_data_jsonl, "r") as f:
        new_examples = [json.loads(line) for line in f if line.strip()]

    with open(old_data_jsonl, "r") as f:
        old_examples = [json.loads(line) for line in f if line.strip()]

    n_replay = int(len(new_examples) * replay_ratio)
    replay_sample = random.sample(old_examples, min(n_replay, len(old_examples)))

    combined = new_examples + replay_sample
    random.shuffle(combined)
    print(
        f"Continual dataset: {len(new_examples)} new + {len(replay_sample)} replay "
        f"= {len(combined)} total"
    )
    return combined


if __name__ == "__main__":
    new_data_path = PROJECT_ROOT / "data" / "processed" / "icd11_update_train.jsonl"
    old_data_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"

    if not new_data_path.exists():
        print(f"⚠️  Future update data not found at {new_data_path}")
        print("Add the file when the next ICD code cycle is released.")
    elif not old_data_path.exists():
        print(f"❌ Base training data not found at {old_data_path}")
    else:
        mixed = create_continual_dataset(new_data_path, old_data_path)
        out_path = PROJECT_ROOT / "data" / "processed" / "continual_train.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for ex in mixed:
                f.write(json.dumps(ex) + "\n")
        print(f"✅ Written {len(mixed)} examples → {out_path}")