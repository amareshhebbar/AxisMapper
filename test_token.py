import os
from dotenv import load_dotenv
from huggingface_hub import whoami
import wandb

load_dotenv(".env")

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    try:
        user_info = whoami(token=hf_token)
        print(f"Hugging Face: Auth successful! Logged in as '{user_info['name']}'")
    except Exception as e:
        print(f"❌ Hugging Face: Auth failed. Invalid token. ({e})")
else:
    print("❌ Hugging Face: HF_TOKEN not found in your .env file.")

wandb_key = os.getenv("WANDB_API_KEY")
if wandb_key:
    try:
        wandb.login(key=wandb_key, verify=True)
        print("Weights & Biases: Auth successful!")
    except Exception as e:
        print(f"❌ W&B: Auth failed. Invalid API key. ({e})")
else:
    print("❌ W&B: WANDB_API_KEY not found in your .env file.")