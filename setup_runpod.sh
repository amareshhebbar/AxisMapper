
set -e

echo "=== Installing packages ==="
pip install --upgrade pip
pip install unsloth
pip install trl peft transformers bitsandbytes accelerate datasets wandb datasketch
pip install flash-attn --no-build-isolation   # takes ~5 min to compile

echo "=== Auth ==="
wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

echo "=== Clone repo ==="
git clone https://github.com/amareshhebbar/AxisMapper /workspace/AxisMapper
cd /workspace/AxisMapper

echo "=== Upload data (run from LOCAL machine) ==="
echo "rsync -av data/processed/ root@$(hostname):/workspace/icd10-finetune/data/processed/"
echo "rsync -av data/raw/icd10cm_codes.json root@$(hostname):/workspace/icd10-finetune/data/raw/"

echo "=== Done ==="