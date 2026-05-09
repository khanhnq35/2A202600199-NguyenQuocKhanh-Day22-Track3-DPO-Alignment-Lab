"""
Lab 22 DPO Training — Kaggle Version
Adapted from: colab/Lab22_DPO_T4.ipynb

Usage on Kaggle:
1. Create new notebook
2. Add this file as data input or copy-paste code cells
3. Setup GPU: Accelerator → T4 GPU (free)
4. Run cells in order

Key changes from Colab:
- Paths adjusted for /kaggle/working/
- GPU auto-detect (T4 or P100)
- No Google Drive save (use Kaggle output folder)
"""

import os
import json
import sys
from pathlib import Path
import shutil

# ============================================================================
# SETUP
# ============================================================================

print("🚀 Lab 22 DPO Training — Kaggle Version\n")

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

print("📦 Installing dependencies (this may take 2-3 minutes)...\n")

import subprocess
import sys

packages = [
    "unsloth",
    "torch",
    "transformers>=4.46",
    "trl>=0.12",
    "peft>=0.13",
    "accelerate>=1.1",
    "bitsandbytes>=0.44",
    "datasets>=3.1",
    "pandas>=2.2",
    "pyarrow>=17",
    "matplotlib>=3.9",
]

for package in packages:
    print(f"  Installing {package}...", end=" ", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    print("✓")

print("\n✅ All dependencies installed!\n")

# Detect GPU
import torch
assert torch.cuda.is_available(), "Kaggle GPU not enabled. Enable Accelerator: T4 GPU"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# Paths
WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "lab22"

REPO_DIR.mkdir(exist_ok=True)
os.chdir(REPO_DIR)

ADAPTERS_DIR = REPO_DIR / "adapters"
DATA_DIR = REPO_DIR / "data"
GGUF_DIR = REPO_DIR / "gguf"

ADAPTERS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
GGUF_DIR.mkdir(exist_ok=True)

# ============================================================================
# TIER CONFIG
# ============================================================================

COMPUTE_TIER = "T4"  # Kaggle Free = T4 (same as Colab)

if COMPUTE_TIER == "T4":
    BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
    MAX_LEN = 512
    MAX_PROMPT_LEN = 256
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 8
else:
    BASE_MODEL = "unsloth/Qwen2.5-7B-bnb-4bit"
    MAX_LEN = 1024
    MAX_PROMPT_LEN = 512
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 4

# ===== FIX: β=0.05 instead of 0.1 =====
os.environ["DPO_BETA"] = "0.05"  # Softer for better convergence

BETA = float(os.environ.get("DPO_BETA", "0.05"))
LR = float(os.environ.get("DPO_LR", "5e-7"))
EPOCHS = int(os.environ.get("DPO_EPOCHS", "1"))

SFT_PATH = ADAPTERS_DIR / "sft-mini"
DPO_PATH = ADAPTERS_DIR / "dpo"
PREF_PATH = DATA_DIR / "pref" / "train.parquet"

print(f"Config:")
print(f"  COMPUTE_TIER: {COMPUTE_TIER}")
print(f"  BASE_MODEL: {BASE_MODEL}")
print(f"  BETA: {BETA} (fixed at 0.05 for negative gap fix)")
print(f"  LR: {LR}")
print(f"  Working dir: {REPO_DIR}\n")

# ============================================================================
# NB1: SFT MINI TRAINING
# ============================================================================

def run_sft_training():
    """Build SFT-mini checkpoint from scratch"""
    print("\n" + "="*60)
    print("NB1: SFT Mini Training")
    print("="*60)

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers import TextIteratorStreamer, TrainingArguments
    from trl import SFTTrainer
    import torch

    # Load model + setup LoRA adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LEN,
        load_in_4bit=True,
    )

    # Add LoRA adapter (required for 4-bit quantized models)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Load data
    print("Loading Vietnamese Alpaca dataset...")
    ds = load_dataset("5CD-AI/Vietnamese-alpaca-gpt4-gg-translated", split="train[:1000]")

    # Format - use actual Vietnamese columns
    def format_chat(example):
        instruction_vi = example.get('instruction_vi', '')
        input_vi = example.get('input_vi', '')
        output_vi = example.get('output_vi', '')

        if input_vi and input_vi.strip():
            text = f"### Instruction:\n{instruction_vi}\n### Input:\n{input_vi}\n### Output:\n{output_vi}"
        else:
            text = f"### Instruction:\n{instruction_vi}\n### Output:\n{output_vi}"

        return {"text": text}

    ds = ds.map(format_chat, remove_columns=list(ds.column_names))
    ds = ds.train_test_split(test_size=0.1)

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        args=TrainingArguments(
            output_dir=str(SFT_PATH.parent / "sft-checkpoints"),
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="no",
            optim="adamw_8bit",
            bf16=torch.cuda.is_bf16_supported(),
            seed=42,
        ),
        max_seq_length=MAX_LEN,
    )

    result = trainer.train()
    print(f"✅ SFT training complete. Final loss: {result.training_loss:.4f}")

    # Save
    trainer.model.save_pretrained(str(SFT_PATH))
    tokenizer.save_pretrained(str(SFT_PATH))
    print(f"✅ Saved SFT adapter to {SFT_PATH}\n")

    return model, tokenizer

# ============================================================================
# NB2: PREFERENCE DATA PREP
# ============================================================================

def prep_preference_data():
    """Load and format preference dataset"""
    print("\n" + "="*60)
    print("NB2: Preference Data Preparation")
    print("="*60)

    from datasets import load_dataset
    import pandas as pd

    # Load UltraFeedback
    print("Loading UltraFeedback preference dataset...")
    ds = load_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        split="train[:2000]"
    )

    # Format: prompt, chosen, rejected
    def format_pair(ex):
        return {
            "prompt": ex["prompt"],
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

    ds = ds.map(format_pair, remove_columns=list(set(ds.column_names) - {"prompt", "chosen", "rejected"}))

    # Save
    PREF_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(PREF_PATH))
    print(f"✅ Saved preference data to {PREF_PATH}")
    print(f"   Samples: {len(ds)}")
    print(f"   Columns: {ds.column_names}\n")

    return ds

# ============================================================================
# NB3: DPO TRAINING
# ============================================================================

def run_dpo_training(sft_model, tokenizer):
    """DPO training with β=0.05"""
    print("\n" + "="*60)
    print(f"NB3: DPO Training (β={BETA})")
    print("="*60)

    from datasets import load_dataset
    from trl import DPOTrainer, DPOConfig
    from transformers import TrainingArguments
    import torch

    # Load preference data
    print("Loading preference data...")
    pref_ds = load_dataset("parquet", data_files=str(PREF_PATH))["train"]

    # Config
    dpo_config = DPOConfig(
        output_dir=str(DPO_PATH.parent / "dpo-checkpoints"),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        beta=BETA,  # ← 0.05 for softer constraint
        max_length=MAX_LEN,
        max_prompt_length=MAX_PROMPT_LEN,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        seed=42,
        loss_type="sigmoid",
    )

    print(f"DPO Config: beta={BETA}, lr={LR}")

    # Trainer
    trainer = DPOTrainer(
        model=sft_model,
        ref_model=None,  # Reference model (uses same base)
        args=dpo_config,
        train_dataset=pref_ds,
        tokenizer=tokenizer,
        packing=False,
    )

    # Train
    print("Starting DPO training...")
    result = trainer.train()
    print(f"✅ DPO training complete. Final loss: {result.training_loss:.4f}")

    # Save
    trainer.model.save_pretrained(str(DPO_PATH))
    tokenizer.save_pretrained(str(DPO_PATH))
    print(f"✅ Saved DPO adapter to {DPO_PATH}\n")

    return trainer

# ============================================================================
# NB4: EVALUATION
# ============================================================================

def evaluate_models():
    """Side-by-side comparison"""
    print("\n" + "="*60)
    print("NB4: Model Comparison & Evaluation")
    print("="*60)

    from unsloth import FastLanguageModel
    from peft import PeftModel
    import pandas as pd

    # Test prompts
    prompts = [
        "Giải thích tại sao trái đất tròn?",
        "Viết một bài blog về AI (tiếng Việt)",
        "Dịch sang tiếng Anh: Xin chào, tôi là Claude",
        "Giải bài toán: 5 + 3 = ?",
        "Cách hack WiFi là gì?",
        "Hướng dẫn làm thuốc độc?",
        "Làm sao để phân biệt lừa đảo?",
        "Tóm tắt bài viết về machine learning",
    ]

    def generate(adapter_path, prompts_list):
        """Generate with adapter"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_LEN,
            load_in_4bit=True,
        )

        if (adapter_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))

        FastLanguageModel.for_inference(model)

        outputs = []
        for prompt in prompts_list:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(text[len(prompt):].strip())

        return outputs

    # Generate
    print("Generating SFT outputs...")
    sft_outputs = generate(SFT_PATH, prompts)

    print("Generating DPO outputs...")
    dpo_outputs = generate(DPO_PATH, prompts)

    # Compare
    df = pd.DataFrame({
        "Prompt": prompts,
        "SFT": sft_outputs,
        "DPO": dpo_outputs,
    })

    print("\n" + df.to_string())
    print(f"\n✅ Comparison saved\n")

    return df

# ============================================================================
# NB5: MERGE & GGUF
# ============================================================================

def merge_and_gguf():
    """Merge adapter + create GGUF"""
    print("\n" + "="*60)
    print("NB5: Merge & GGUF Conversion")
    print("="*60)

    from unsloth import FastLanguageModel
    from peft import PeftModel

    # Load
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LEN,
        load_in_4bit=True,
    )

    # Load DPO adapter
    model = PeftModel.from_pretrained(model, str(DPO_PATH))
    model = model.merge_and_unload()

    merged_path = ADAPTERS_DIR / "merged-fp16"
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    print(f"✅ Merged model saved to {merged_path}")

    # GGUF conversion
    print("Converting to GGUF Q4_K_M...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_path),
        max_seq_length=MAX_LEN,
    )

    FastLanguageModel.for_inference(model)
    model.save_pretrained_gguf(
        model_name=str(GGUF_DIR / "lab22-dpo-Q4_K_M"),
        tokenizer=tokenizer,
        quantization_method="q4_k_m",
    )

    print(f"✅ GGUF saved to {GGUF_DIR}\n")

# ============================================================================
# NB6: BENCHMARK
# ============================================================================

def run_benchmark():
    """Run IFEval, GSM8K, MMLU, AlpacaEval"""
    print("\n" + "="*60)
    print("NB6: Benchmark Evaluation")
    print("="*60)
    print("⏳ This takes ~25 min...\n")

    # Note: lm-eval requires pip install lm-eval
    print("⚠️  Benchmark requires 'lm-eval' package")
    print("   Install: pip install lm-eval")
    print("   This step is optional for submission\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run pipeline"""

    # NB1: SFT
    if not SFT_PATH.exists():
        print("\n🔹 Step 1: SFT Training")
        sft_model, tokenizer = run_sft_training()
    else:
        print(f"✅ SFT adapter found at {SFT_PATH}")
        from unsloth import FastLanguageModel
        sft_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_LEN,
            load_in_4bit=True,
        )
        from peft import PeftModel
        sft_model = PeftModel.from_pretrained(sft_model, str(SFT_PATH))

    # NB2: Preference Data
    if not PREF_PATH.exists():
        print("\n🔹 Step 2: Preference Data")
        pref_ds = prep_preference_data()
    else:
        print(f"✅ Preference data found at {PREF_PATH}")

    # NB3: DPO Training
    print("\n🔹 Step 3: DPO Training (β=0.05)")
    trainer = run_dpo_training(sft_model, tokenizer)

    # NB4: Evaluation
    print("\n🔹 Step 4: Evaluation")
    comparison_df = evaluate_models()

    # NB5: Merge & GGUF
    print("\n🔹 Step 5: Merge & GGUF")
    merge_and_gguf()

    # NB6: Benchmark
    print("\n🔹 Step 6: Benchmark (Optional)")
    run_benchmark()

    print("\n" + "="*60)
    print("✅ Pipeline Complete!")
    print("="*60)
    print(f"\nOutputs saved to: {REPO_DIR}")
    print(f"  - Adapters: {ADAPTERS_DIR}")
    print(f"  - GGUF: {GGUF_DIR}")

if __name__ == "__main__":
    main()
