# Colab Fix Guide — Lab 22 DPO

## Problem
- ❌ Reward gap went **NEGATIVE** (-0.240) → DPO learned wrong direction
- ❌ Benchmark (NB6) not run yet
- ✅ REFLECTION.md updated with analysis

## Solution
1. **Re-run NB3 (DPO) với β = 0.05** (softer constraint)
2. **Run NB6 (Benchmark)** để hoàn thành
3. **Re-generate comparison plot** từ new DPO outputs

---

## Step-by-Step Instructions

### 🔴 **Step 1: Open Colab Notebook**
- Go to: https://colab.research.google.com/github/<your-username>/Day22-Track3-DPO-Alignment-Lab/blob/main/colab/Lab22_DPO_T4.ipynb
- Runtime → Change runtime type → **T4 GPU** → Connect

### 🔴 **Step 2: Find & Replace Cell 54 (DPO Config)**

**FIND THIS CELL** (tìm ô chứa `DPOConfig`):
```python
from trl import DPOConfig

dpo_config = DPOConfig(
    output_dir=str(DPO_OUT.parent / "dpo-checkpoi...
```

**REPLACE với code dưới đây:**

```python
from trl import DPOConfig

# === FIX: Try β = 0.05 (softer) instead of 0.1 ===
# Root cause of negative gap: likely data signal issue, not hyperparameter
# Lower β = softer constraint = more flexible learning

dpo_config = DPOConfig(
    output_dir=str(DPO_OUT.parent / "dpo-checkpoints"),
    
    # ===== CRITICAL CHANGE =====
    beta=0.05,  # WAS: 0.1 (TOO AGGRESSIVE) → NOW: 0.05 (SOFTER)
    # ===========================
    
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",
    save_strategy="no",
    logging_steps=10,
    label_smoothing=0.0,
    loss_type="sigmoid",
    remove_unused_columns=False,
)

print(f"DPOConfig: beta={dpo_config.beta}  lr={dpo_config.learning_rate}  loss_type={dpo_config.loss_type}")
```

### 🔴 **Step 3: Run Cells in Order**

**Clear previous adapters** (before NB3):
```python
import shutil

# Clear old dpo adapter to start fresh
if DPO_OUT.exists():
    shutil.rmtree(DPO_OUT)
    print(f"Cleared {DPO_OUT}")

DPO_OUT.mkdir(parents=True, exist_ok=True)
print(f"Created fresh {DPO_OUT}")
```

**Then run**:
1. Cell 54 (modified DPOConfig with β=0.05)
2. Cell 58 (DPOTrainer init)
3. Cell 59 (trainer.train()) — **WAIT ~18 min**
4. Cell 61-65 (reward curves + save) — **CRITICAL: Check if reward gap is positive now**

### 🔴 **Step 4: Check Reward Curves**

After Cell 63 runs, **verify output**:

Expected (✅ GOOD):
```
END  chosen reward:    -0.3XX  (hoặc dương)
END  rejected reward:  -0.5XX  (hoặc thấp hơn)
END  reward gap:       +0.1XX  (POSITIVE! 👍)
```

If still negative ❌:
- Data issue confirmed → skip to Step 5 (run benchmark with current SFT)
- Hoặc try β = 0.02 more aggressively

### 🔴 **Step 5: Generate New Comparison**

After new DPO saves, **re-run Cells 73-86** (NB4):
```
Cell 73: Load base model
Cell 75: Generate with SFT-only
Cell 77: Generate with NEW SFT+DPO (β=0.05)
Cell 79: Build comparison table
Cell 81: Plot side-by-side
Cell 83-86: Judge + win/loss/tie count
```

Expected improvement:
- From: 0/8 wins → To: ~2-4/8 wins (if β=0.05 helps)
- Or stay tied 8/8 (if data is truly swapped)

### 🔴 **Step 6: Run Benchmark (NB6)**

**Find cells starting with**:
```python
print(">>> SFT-only on IFEval")
sft_ifeval = run_lm_eval(SFT_PATH, "ifeval", ...
```

**Uncomment and run**:
```python
print(">>> SFT-only on IFEval")
sft_ifeval = run_lm_eval(SFT_PATH, "ifeval", LIMIT_IFEVAL, num_fewshot=0, label="SFT")

print(">>> SFT-only on GSM8K")
sft_gsm8k = run_lm_eval(SFT_PATH, "gsm8k", LIMIT_GSM8K, num_fewshot=8, label="SFT")

print(">>> SFT-only on MMLU (sampled)")
sft_mmlu = run_lm_eval(SFT_PATH, "mmlu", LIMIT_MMLU, num_fewshot=0, label="SFT")

# === AFTER SFT, RUN DPO VERSIONS ===
print("\n>>> SFT+DPO on IFEval")
dpo_ifeval = run_lm_eval(DPO_PATH, "ifeval", LIMIT_IFEVAL, num_fewshot=0, label="DPO")

print(">>> SFT+DPO on GSM8K")
dpo_gsm8k = run_lm_eval(DPO_PATH, "gsm8k", LIMIT_GSM8K, num_fewshot=8, label="DPO")

print(">>> SFT+DPO on MMLU (sampled)")
dpo_mmlu = run_lm_eval(DPO_PATH, "mmlu", LIMIT_MMLU, num_fewshot=0, label="DPO")
```

**Then run the plot cell** (final bar chart).

---

## ⏱️ **Estimated Time**
- Cell 54-65 (new DPO): ~18 min
- Cell 73-86 (new comparison): ~3 min
- Cell 115-132 (benchmark): ~25 min
- **Total**: ~50 min on free T4

---

## 📋 **Checklist Before Submit**

- [ ] Modified Cell 54 with β=0.05
- [ ] New DPO training ran (Cell 59)
- [ ] Reward gap checked - report if positive/negative in chat
- [ ] New comparison table generated (Cell 81)
- [ ] Win/loss counts updated (Cell 86)
- [ ] Benchmark completed (Cells 120-132)
- [ ] Benchmark plot generated
- [ ] `data/eval/benchmark_results.json` exists
- [ ] All output cells preserved in notebook
- [ ] REFLECTION.md updated ✅ (done)
- [ ] Run `make verify` locally (or check artifacts exist)

---

## 🆘 **Troubleshooting**

### If reward gap STILL negative after β=0.05:
```python
# Try MUCH softer β
beta=0.02  # Even softer

# Or check if data is swapped:
# Print first 3 examples from preference data
from datasets import load_from_disk
pref_data = load_from_disk(str(PREF_DATA_DIR / "train"))
for i in range(3):
    ex = pref_data[i]
    print(f"\n[{i}] Prompt: {ex['prompt'][:100]}")
    print(f"  Chosen:  {ex['chosen'][:100]}")
    print(f"  Rejected: {ex['rejected'][:100]}")
    # Manually: does "chosen" look better than "rejected"?
```

### If benchmark runs too slow:
```python
# Reduce sample sizes
LIMIT_IFEVAL = 200  # was 500
LIMIT_GSM8K = 100   # was 500
LIMIT_MMLU = 200    # was 500
```

### If Colab times out:
- Save outputs to Google Drive
- Restart runtime
- Re-run from checkpoint

---

## 📸 **Screenshots to Capture**

After all cells run:

1. ✅ Cell 63 output → `03-dpo-reward-curves.png` (check gap sign)
2. ✅ Cell 81 output → `04-side-by-side-table.png`
3. ✅ Cell 86 output (win/loss/tie) → `04b-judge-summary.png`
4. ✅ Cell 132 output (benchmark plot) → `07-benchmark-comparison.png`

Copy these to `submission/screenshots/` folder locally.

---

## 📧 **Need Help?**

- If reward gap still negative → data is likely swapped/bad → mention in REFLECTION
- If benchmark runs out of memory → reduce LIMIT_*
- If any cell fails → read error message + check your COMPUTE_TIER
