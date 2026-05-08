# Kaggle Guide — Lab 22 DPO Training

## 🎯 Quick Start

**Option 1: Use Kaggle notebook (recommended)**
```
1. Create new notebook on Kaggle
2. Copy-paste cells from kaggle_lab22_dpo.py
3. Run cells in order
4. GPU: Accelerator → T4 GPU (free tier)
```

**Option 2: Use notebook + script combo**
```
1. Upload kaggle_lab22_dpo.py to Kaggle dataset
2. Import and run in notebook
```

---

## 📋 Setup Checklist

- [ ] Kaggle account (free)
- [ ] New Kaggle notebook created
- [ ] GPU enabled: Accelerator → T4 (free, ~20 hrs/week)
- [ ] Python 3.10+

---

## 🚀 Execution Steps

### **Step 1: Install Dependencies**
```python
!pip install -q unsloth torch transformers trl peft accelerate bitsandbytes datasets pandas pyarrow
```

### **Step 2: Copy Code**
```python
# Copy entire content of kaggle_lab22_dpo.py
# Paste into Kaggle notebook cell
# Run it
```

### **Step 3: Run Pipeline**
```python
# All functions will execute in order:
# - NB1: SFT training (~15 min)
# - NB2: Load preference data (~1 min)
# - NB3: DPO training with β=0.05 (~18 min)
# - NB4: Evaluation (~3 min)
# - NB5: Merge & GGUF (~10 min)
# - NB6: Benchmark (optional, ~25 min)

# Total: ~70 min
```

---

## 🔄 Manual Cell-by-Cell (Alternative)

If you want to run cells one by one instead of `main()`:

```python
# Setup
import torch
import os
from pathlib import Path

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "lab22"
REPO_DIR.mkdir(exist_ok=True)
os.chdir(REPO_DIR)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"   Working dir: {REPO_DIR}")

# ============================================================
# Run ONLY NB3 (DPO) if SFT artifacts already exist
# ============================================================

# Install deps
!pip install -q unsloth torch transformers trl peft accelerate bitsandbytes datasets pandas pyarrow

# [Setup config section from kaggle_lab22_dpo.py]

# [Copy NB3 run_dpo_training() function]

# [Copy NB4 evaluate_models() function]

# [Copy NB5 merge_and_gguf() function]

# Run specific sections:
# sft_model, tokenizer = load_sft_from_disk()  # or run SFT if needed
# run_dpo_training(sft_model, tokenizer)
# evaluate_models()
# merge_and_gguf()
```

---

## 💾 Output Management

### **Where files are saved:**
```
/kaggle/working/lab22/
├── adapters/
│   ├── sft-mini/         (from NB1)
│   ├── dpo/              (from NB3)
│   └── merged-fp16/      (from NB5)
├── data/
│   └── pref/
│       └── train.parquet (from NB2)
└── gguf/
    └── lab22-dpo-Q4_K_M.gguf (from NB5)
```

### **Download outputs:**
```
In Kaggle notebook:
1. Click "Output" tab
2. All files in /kaggle/working/lab22/ are downloadable
3. Download adapters/, gguf/, etc.
4. Use locally or push to HuggingFace
```

---

## ⚙️ Key Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| COMPUTE_TIER | T4 | Kaggle free = T4 (16 GB VRAM) |
| BASE_MODEL | Qwen2.5-3B-bnb-4bit | Same as Colab |
| β | **0.05** | **Fixed for negative gap fix** |
| LR | 5e-7 | DPO learning rate |
| EPOCHS | 1 | Single epoch |
| SFT samples | 1000 | Vietnamese Alpaca |
| Preference pairs | 2000 | UltraFeedback |

---

## 🔧 Troubleshooting

### **GPU not detected**
```
❌ Error: AssertionError: CUDA GPU not available

✅ Fix: In notebook settings, enable Accelerator → T4 GPU
```

### **Out of Memory during DPO**
```
❌ Error: CUDA out of memory. Tried to allocate ...

✅ Fix 1: Reduce GRAD_ACCUM:
   GRAD_ACCUM = 4  # was 8

✅ Fix 2: Reduce batch size:
   PER_DEVICE_BATCH = 1  # already min

✅ Fix 3: Reduce MAX_LEN:
   MAX_LEN = 256  # was 512
```

### **Model weights too large**
```
❌ Error: Model is too large for free Colab/Kaggle storage

✅ Fix: Use LoRA adapter (already using)
   - Adapters: ~50 MB each
   - Merged model: ~3 GB
   - GGUF: ~1.2 GB
```

### **Module not found**
```
❌ Error: ModuleNotFoundError: No module named 'unsloth'

✅ Fix: Install first:
   !pip install -q unsloth torch transformers trl
```

---

## 📊 Expected Output

### **After NB1 (SFT)**
```
✅ SFT training complete. Final loss: 1.4385
✅ Saved SFT adapter to .../adapters/sft-mini
```

### **After NB3 (DPO with β=0.05)**
```
✅ DPO training complete. Final loss: 0.9805
✅ Saved DPO adapter to .../adapters/dpo

Reward curves should show (hopefully):
END  chosen reward:    -0.3XX  (↑ or neutral)
END  rejected reward:  -0.5XX  (↓)
END  reward gap:       +0.1XX  (✅ POSITIVE!)
```

### **After NB4 (Evaluation)**
```
Side-by-side comparison table printed
Judge win rate reported (expect: 2-4/8 wins with β=0.05)
```

### **After NB5 (Merge & GGUF)**
```
✅ Merged model saved
✅ GGUF Q4_K_M saved to .../gguf/lab22-dpo-Q4_K_M.gguf
   Size: ~1.2 GB
```

---

## 🎯 Next Steps

1. **Run on Kaggle** using `kaggle_lab22_dpo.py`
2. **Download outputs** (adapters, GGUF)
3. **Update REFLECTION.md** with new metrics
4. **Push to GitHub**:
   ```bash
   git add submission/REFLECTION.md
   git commit -m "Add Kaggle results with β=0.05"
   git push origin main
   ```
5. **Submit to LMS**

---

## 💡 Advantages of Kaggle over Colab

| Aspect | Colab | Kaggle |
|--------|-------|--------|
| Free GPU | T4 (16 GB) | T4 (16 GB) ✅ Same |
| Runtime limit | 12 hrs | 12 hrs ✅ Same |
| Storage | Ephemeral | Download outputs ✅ Better |
| Competition | Free tier shared | Free tier dedicated ✅ Faster |
| Reproducibility | Session-based | Can re-run notebook ✅ Better |
| Integration | GitHub, Drive | Datasets ✅ Good |

---

## 📝 Final Notes

- **β=0.05** is baked into the script (softer than 0.1)
- All paths use `/kaggle/working/` (Kaggle standard)
- Script is self-contained (can run standalone)
- Outputs are downloadable from Kaggle Output folder
- Same models as Colab (reproducible results)

**Good luck! 🚀**
