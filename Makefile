PYTHON      := python3
PIP         := pip3
ROOT        := $(shell pwd)

RAW_DIR          := $(ROOT)/data/raw
PROCESSED_DIR    := $(ROOT)/data/processed
DPO_DIR          := $(ROOT)/data/dpo

XML_FILE         := $(RAW_DIR)/icd10cm_tabular_2026.xml
CODES_JSON       := $(RAW_DIR)/icd10cm_codes_2026.json
RAW_SYNTHETIC    := $(RAW_DIR)/raw_synthetic.jsonl

TRAIN_JSONL      := $(PROCESSED_DIR)/train.jsonl
VAL_JSONL        := $(PROCESSED_DIR)/val.jsonl
TEST_JSONL       := $(PROCESSED_DIR)/test.jsonl
ORPO_TRAIN       := $(DPO_DIR)/orpo_train.jsonl
ORPO_VAL         := $(DPO_DIR)/orpo_val.jsonl

GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
CYAN   := \033[0;36m
RESET  := \033[0m

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo ""
	@echo "$(CYAN)AxisMapper — ICD-10 Fine-tuning Pipeline$(RESET)"
	@echo "────────────────────────────────────────────────"
	@echo "  $(GREEN)make setup$(RESET)          install all dependencies"
	@echo "  $(GREEN)make dataset$(RESET)        parse XML + build all JSONL splits"
	@echo "  $(GREEN)make check-dataset$(RESET)  verify every data file is present"
	@echo "  $(GREEN)make train-local$(RESET)    smoke-test (20 steps, 1.5B model)"
	@echo "  $(GREEN)make train-runpod$(RESET)   full QLoRA SFT  (RunPod / cloud)"
	@echo "  $(GREEN)make orpo$(RESET)           ORPO preference alignment"
	@echo "  $(GREEN)make eval$(RESET)           evaluate on test split"
	@echo "  $(GREEN)make merge$(RESET)          merge adapter → fp16 + GGUF"
	@echo "  $(GREEN)make continual$(RESET)      continual learning (ICD update)"
	@echo "  $(GREEN)make all-runpod$(RESET)     train → orpo → eval → merge"
	@echo "  $(GREEN)make clean$(RESET)          delete outputs/"
	@echo "  $(GREEN)make help$(RESET)           show this message"
	@echo ""

.PHONY: setup
setup:
	@echo "$(CYAN)=== [1/3] Upgrading pip ===$(RESET)"
	$(PIP) install --upgrade pip

	@echo "$(CYAN)=== [2/3] Installing core ML stack ===$(RESET)"
	$(PIP) install -r $(ROOT)/requirements.txt

	@echo "$(CYAN)=== [3/3] Verifying auth tokens ===$(RESET)"
	$(PYTHON) $(ROOT)/test_token.py

	@echo "$(GREEN)✅  setup complete$(RESET)"

.PHONY: dataset
dataset: _check-xml _parse-xml _check-processed _build-orpo check-dataset
	@echo "$(GREEN)✅  All datasets ready$(RESET)"

.PHONY: _check-xml
_check-xml:
	@echo "$(CYAN)=== Checking for raw ICD-10-CM XML ===$(RESET)"
	@if [ ! -f "$(XML_FILE)" ]; then \
		echo "$(RED)❌  Missing: $(XML_FILE)$(RESET)"; \
		echo ""; \
		echo "Download from:"; \
		echo "  https://www.cms.gov/medicare/coding-billing/icd-10-codes"; \
		echo "  → April 1, 2026 Code Tables, Tabular and Index (ZIP)"; \
		echo "  → extract icd10cm_tabular_2026.xml → place at data/raw/"; \
		echo ""; \
		exit 1; \
	fi
	@echo "$(GREEN)✅  XML found$(RESET)"

.PHONY: _parse-xml
_parse-xml:
	@echo "$(CYAN)=== Parsing ICD-10-CM XML → JSON ===$(RESET)"
	@mkdir -p $(RAW_DIR)
	$(PYTHON) $(ROOT)/scripts/parse_icd10_xml.py
	@echo "$(GREEN)✅  $(CODES_JSON) written$(RESET)"

.PHONY: _check-processed
_check-processed:
	@echo "$(CYAN)=== Checking processed JSONL splits ===$(RESET)"
	@for f in "$(TRAIN_JSONL)" "$(VAL_JSONL)" "$(TEST_JSONL)"; do \
		if [ ! -f "$$f" ]; then \
			echo "$(RED)❌  Missing: $$f$(RESET)"; \
			echo "Run:  git lfs pull   (or copy your JSONL files here)"; \
			exit 1; \
		fi; \
		lines=$$(wc -l < "$$f"); \
		if [ "$$lines" -lt 2 ]; then \
			echo "$(RED)❌  $$f looks like an LFS pointer ($$lines lines). Run: git lfs pull$(RESET)"; \
			exit 1; \
		fi; \
		echo "$(GREEN)✅  $$f  ($$lines examples)$(RESET)"; \
	done

.PHONY: _build-orpo
_build-orpo:
	@echo "$(CYAN)=== Building ORPO preference pairs ===$(RESET)"
	@if [ ! -f "$(RAW_SYNTHETIC)" ]; then \
		echo "$(RED)❌  Missing: $(RAW_SYNTHETIC)$(RESET)"; \
		echo "Place your raw synthetic JSONL at data/raw/raw_synthetic.jsonl"; \
		exit 1; \
	fi
	@mkdir -p $(DPO_DIR)
	$(PYTHON) $(ROOT)/scripts/build_orpo_dataset.py
	@echo "$(GREEN)✅  ORPO splits written to data/dpo/$(RESET)"

.PHONY: check-dataset
check-dataset:
	@echo ""
	@echo "$(CYAN)=== Dataset Health Check ===$(RESET)"
	@PASS=1; \
	for f in \
		"$(CODES_JSON)" \
		"$(TRAIN_JSONL)" \
		"$(VAL_JSONL)" \
		"$(TEST_JSONL)" \
		"$(ORPO_TRAIN)" \
		"$(ORPO_VAL)"; \
	do \
		if [ ! -f "$$f" ]; then \
			echo "$(RED)  MISSING  $$f$(RESET)"; PASS=0; \
		else \
			lines=$$(wc -l < "$$f"); \
			size=$$(du -sh "$$f" | cut -f1); \
			if [ "$$lines" -lt 2 ]; then \
				echo "$(RED)  LFS PTR  $$f  (run: git lfs pull)$(RESET)"; PASS=0; \
			else \
				echo "$(GREEN)  OK       $$f  [$$lines lines, $$size]$(RESET)"; \
			fi; \
		fi; \
	done; \
	echo ""; \
	if [ "$$PASS" -eq 0 ]; then \
		echo "$(RED)❌  Some files are missing or unresolved LFS pointers.$(RESET)"; \
		exit 1; \
	else \
		echo "$(GREEN)✅  All dataset files healthy$(RESET)"; \
	fi

.PHONY: train-local
train-local:
	@echo "$(CYAN)=== Smoke-test Training (20 steps, Qwen2.5-1.5B) ===$(RESET)"
	@echo "$(YELLOW)→ Uses sft_runpod.py with SMOKE_TEST=1 env var (20 examples, 20 steps, no W&B)$(RESET)"
	SMOKE_TEST=1 $(PYTHON) $(ROOT)/train/sft_local.py
	@echo "$(GREEN)✅  Local smoke-test passed$(RESET)"

.PHONY: train-runpod
train-runpod: check-dataset _check-env
	@echo "$(CYAN)=== Full QLoRA SFT — Qwen2.5-7B (RunPod) ===$(RESET)"
	$(PYTHON) $(ROOT)/train/sft_runpod.py
	@echo "$(GREEN)✅  SFT training complete$(RESET)"

.PHONY: orpo
orpo: check-dataset _check-env
	@echo "$(CYAN)=== ORPO Preference Alignment ===$(RESET)"
	$(PYTHON) $(ROOT)/train/orpo_train.py
	@echo "$(GREEN)✅  ORPO alignment complete$(RESET)"

.PHONY: eval
eval: check-dataset _check-hf-token
	@echo "$(CYAN)=== Evaluation on Test Split ===$(RESET)"
	PYTHONPATH=$(ROOT) $(PYTHON) $(ROOT)/eval/evaluate.py
	@echo "$(GREEN)✅  Evaluation complete$(RESET)"

.PHONY: merge
merge: _check-hf-token
	@echo "$(CYAN)=== Merging Adapter + Exporting GGUF ===$(RESET)"
	$(PYTHON) $(ROOT)/train/merge_export.py
	@echo "$(GREEN)✅  Merge + export complete$(RESET)"

.PHONY: continual
continual: _check-env
	@echo "$(CYAN)=== Continual Learning Pass ===$(RESET)"
	$(PYTHON) $(ROOT)/train/sft_continual.py
	@echo "$(GREEN)✅  Continual training complete$(RESET)"

.PHONY: all-runpod
all-runpod:
	@echo ""
	@echo "$(CYAN)╔══════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║   AxisMapper — Full RunPod Pipeline      ║$(RESET)"
	@echo "$(CYAN)╚══════════════════════════════════════════╝$(RESET)"
	@echo ""

	@echo "$(YELLOW)[1/5] Dataset check$(RESET)"
	@$(MAKE) check-dataset

	@echo ""
	@echo "$(YELLOW)[2/5] QLoRA SFT — Qwen2.5-7B$(RESET)"
	@$(MAKE) train-runpod

	@echo ""
	@echo "$(YELLOW)[3/5] ORPO preference alignment$(RESET)"
	@$(MAKE) orpo

	@echo ""
	@echo "$(YELLOW)[4/5] Evaluation$(RESET)"
	@$(MAKE) eval

	@echo ""
	@echo "$(YELLOW)[5/5] Merge + export$(RESET)"
	@$(MAKE) merge

	@echo ""
	@echo "$(GREEN)╔══════════════════════════════════════════╗$(RESET)"
	@echo "$(GREEN)║   ✅  Full pipeline complete!             ║$(RESET)"
	@echo "$(GREEN)╚══════════════════════════════════════════╝$(RESET)"
	@echo ""


.PHONY: clean
clean:
	@echo "$(YELLOW)=== Removing outputs/ ===$(RESET)"
	rm -rf $(ROOT)/outputs
	@echo "$(GREEN)✅  Cleaned$(RESET)"

.PHONY: _check-env
_check-env: _check-hf-token _check-wandb

.PHONY: _check-hf-token
_check-hf-token:
	@if [ -z "$$HF_TOKEN" ] && [ ! -f "$(ROOT)/.env" ]; then \
		echo "$(RED)❌  HF_TOKEN not set and no .env file found$(RESET)"; \
		echo "Create a .env file with HF_TOKEN=hf_... or export it before running make"; \
		exit 1; \
	fi

.PHONY: _check-wandb
_check-wandb:
	@if [ -z "$$WANDB_API_KEY" ] && [ ! -f "$(ROOT)/.env" ]; then \
		echo "$(RED)❌  WANDB_API_KEY not set and no .env file found$(RESET)"; \
		echo "Create a .env file with WANDB_API_KEY=... or export it before running make"; \
		exit 1; \
	fi