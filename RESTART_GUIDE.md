# Restart Guide — Lab 22 (Sau khi end session cũ)

## 📝 **Status Hiện Tại**

✅ **Đã có**:
- SFT training (NB1) ✓
- Preference data (NB2) ✓
- DPO training (NB3) — nhưng β=0.1 (cần re-run với β=0.05)
- GGUF merge (NB5) ✓
- REFLECTION.md ✓
- Notebook sửa sẵn (β override cell)

❌ **Chưa có**:
- DPO training với β=0.05 (cần chạy lại)
- Evaluation mới (NB4) (cần chạy lại)
- Benchmark (NB6) (chưa chạy)
- Screenshots mới (cần capture)

---

## 🚀 **Steps to Resume**

### **Step 1: Open Colab Notebook**
```
https://colab.research.google.com/github/khanhnq35/2A202600199-NguyenQuocKhanh-Day22-Track3-DPO-Alignment-Lab/blob/main/colab/Lab22_DPO_T4.ipynb
```

### **Step 2: Clear Old Artifacts (Optional but Recommended)**
```python
# Run this cell to clear old outputs
import shutil

paths_to_clear = [
    "/content/lab22/adapters/dpo",
    "/content/lab22/data/eval",
]

for path in paths_to_clear:
    if shutil.os.path.exists(path):
        shutil.rmtree(path)
        print(f"✓ Cleared {path}")
```

### **Step 3: Run Notebook from Start**

**Option A: Run ALL from beginning** (safest):
```
Runtime → Run all (Ctrl+F9)
```

**Option B: Run specific sections only** (faster):
```
1. Cells 1-25:   SFT training (NB1) — can skip if artifacts exist
2. Cells 26-45:  Preference data (NB2) — can skip
3. Cells 47-65:  DPO training (NB3) ← RUN THIS (β=0.05 override cell is here)
4. Cells 75-86:  Re-evaluate (NB4)
5. Cells 120-132: Benchmark (NB6)
```

---

## ⏱️ **Timeline (Recommended Path)**

### **Path: Re-run only NB3 + NB4 + NB6** (~50 min)

```
1. Open Colab notebook
2. Scroll to cell with "===== FIX: Override β =====" comment
3. Run cell (β override)
4. Run Cell 54 (DPOConfig) — now uses β=0.05
5. Run Cell 58 (DPOTrainer)
6. Run Cell 59 (trainer.train()) ⏳ ~18 min — WAIT HERE
7. Run Cell 61-63 (check reward gap)
   - Look for "reward gap:" line
   - ✅ If positive: continue
   - ❌ If still negative: try β=0.02
8. Run Cell 75-86 (new evaluation)
9. Run Cell 120-132 (benchmark) ⏳ ~25 min — WAIT HERE
10. Screenshot key outputs
11. Update REFLECTION.md §2 + §7
12. Push to GitHub
```

**Total**: ~50 min (mostly waiting for training)

---

## 📋 **Checklist: What to Run**

### **Essential** (MUST RUN):

- [ ] **β override cell** (new cell before DPOConfig)
  - Sets `os.environ['DPO_BETA'] = '0.05'`
  
- [ ] **NB3 Training** (Cell 54-65)
  - Cell 54: DPOConfig (uses β=0.05 from override)
  - Cell 58: DPOTrainer init
  - Cell 59: trainer.train() ← **18 min**
  - Cell 61-63: Reward curves + check gap sign
  
- [ ] **NB4 Evaluation** (Cell 75-86)
  - Cell 75: Generate SFT outputs
  - Cell 77: Generate DPO outputs (β=0.05)
  - Cell 79-81: Build comparison table
  - Cell 83-86: Judge + win/loss count
  
- [ ] **NB6 Benchmark** (Cell 120-132)
  - Cell 120-127: SFT benchmarks
  - Cell 129-132: DPO benchmarks + plot
  
- [ ] **Update REFLECTION.md**
  - §2: New metrics from β=0.05 run
  - §7: Benchmark results
  
- [ ] **Capture Screenshots**
  - `03-dpo-reward-curves.png` (Cell 61)
  - `04-side-by-side-table.png` (Cell 81)
  - `04b-judge-summary.png` (Cell 86)
  - `07-benchmark-comparison.png` (Cell 132)

### **Optional** (CAN SKIP):

- [ ] NB1 SFT training (can skip if artifacts exist)
- [ ] NB2 Preference data (can skip if parquet exists)
- [ ] NB5 Merge & GGUF (skip unless need new GGUF)

---

## 🔍 **Key Cells to Watch**

### **Cell 63 Output** (CRITICAL)
```
✅ Good:
END  chosen reward:    -0.3XX
END  rejected reward:  -0.5XX
END  reward gap:       +0.1XX  ← POSITIVE!

❌ Bad:
END  chosen reward:    -0.8XX
END  rejected reward:  -0.6XX
END  reward gap:       -0.2XX  ← NEGATIVE!
```

### **Cell 86 Output** (Judge Summary)
```
============================================================
WIN/LOSS/TIE SUMMARY (8 prompts)
============================================================
Overall:        SFT-only: X/8   SFT+DPO: X/8   tie: X/8
```

Expected improvement from β=0.05:
- Before: 0/8 wins (all ties)
- After: 2-4/8 wins (improved)

### **Cell 132 Output** (Benchmark Plot)
```
4-bar chart with:
- IFEval: SFT vs DPO
- GSM8K: SFT vs DPO
- MMLU: SFT vs DPO
- AlpacaEval: SFT vs DPO
```

---

## 💾 **Files to Update After Running**

1. **submission/REFLECTION.md**
   - §2: Update with new metrics
   - §3: Reward curves analysis (check if gap is now positive)
   - §7: Benchmark results + interpretation

2. **submission/screenshots/**
   - Add 4 new PNGs (03, 04, 04b, 07)

3. **GitHub push**
   ```bash
   git add submission/
   git commit -m "Lab 22: Add results from β=0.05 re-run"
   git push origin main
   ```

---

## 🎯 **Final Deliverables**

```
submission/
├── REFLECTION.md ✅ (updated)
└── screenshots/
    ├── 01-setup-gpu.png ✅ (existing)
    ├── 02-sft-loss.png ✅ (existing)
    ├── 03-dpo-reward-curves.png ← NEW
    ├── 04-side-by-side-table.png ← NEW
    ├── 05-judge-output.png ✅ (if using judge)
    ├── 06-gguf-smoke.png ✅ (existing)
    └── 07-benchmark-comparison.png ← NEW
```

---

## 🚨 **If Something Goes Wrong**

### **Reward gap STILL negative after β=0.05:**
```python
# Try even softer β
os.environ['DPO_BETA'] = '0.02'

# OR accept that data is bad and mention in REFLECTION §3:
# "UltraFeedback English preference pairs do not align with 
#  Qwen-3B Vietnamese. Even with β=0.02, reward gap remained negative.
#  Likely requires Vietnamese-specific preference data."
```

### **Out of memory during benchmark:**
```python
# Reduce sampling limits before running benchmark cells
LIMIT_IFEVAL = 200   # was 500
LIMIT_GSM8K = 100    # was 500
LIMIT_MMLU = 200     # was 500
```

### **Colab times out:**
- Save outputs to Google Drive periodically
- Restart runtime if needed
- Re-run from last successful cell

---

## 📞 **Summary**

**To continue after session restart:**

1. ✅ Open Colab notebook (already has β override cell)
2. ✅ Run β override cell (before Cell 54)
3. ✅ Run NB3 training (Cell 54-65) with β=0.05
4. ✅ Run NB4 evaluation (Cell 75-86)
5. ✅ Run NB6 benchmark (Cell 120-132)
6. ✅ Update REFLECTION.md
7. ✅ Capture 4 screenshots
8. ✅ Push to GitHub

**Total time**: ~50 min (mostly waiting)

**Next deadline**: Submit to LMS before 23:59 tomorrow

Good luck! 🚀
