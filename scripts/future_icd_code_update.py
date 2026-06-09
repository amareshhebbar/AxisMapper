import json
import random

def create_continual_dataset(new_data_jsonl: str, old_data_jsonl: str, replay_ratio: float = 0.08) -> list[dict]:
    with open(new_data_jsonl, "r") as f:
        new_examples = [json.loads(line) for line in f if line.strip()]
        
    with open(old_data_jsonl, "r") as f:
        old_examples = [json.loads(line) for line in f if line.strip()]
    
    n_replay = int(len(new_examples) * replay_ratio)
    replay_sample = random.sample(old_examples, min(n_replay, len(old_examples)))
    
    combined = new_examples + replay_sample
    random.shuffle(combined)
    return combined