# 🚀 Quick Fix — Lab 22 (TL;DR)

## What's Done ✅
- ✅ SFT training (NB1)
- ✅ Preference data prep (NB2)
- ✅ DPO training (NB3) — **but with NEGATIVE reward gap ⚠️**
- ✅ GGUF merge (NB5)
- ✅ REFLECTION.md updated with analysis

## What's Broken ❌
- ❌ Reward gap = -0.240 (should be positive)
  - Chosen reward: -0.863
  - Rejected reward: -0.623 ← **HIGHER than chosen** (backwards!)
- ❌ All 8 test prompts = TIE (0 wins for SFT+DPO)
- ❌ Benchmark (NB6) not run yet

## The Fix 🔧

### 1. **Change β from 0.1 → 0.05**
   - Root cause: UltraFeedback English data doesn't align with Qwen-3B Vietnamese
   - Softer β (0.05) = more flexible learning

### 2. **Re-run DPO training** with new β
   - ~18 min on free T4

### 3. **Run benchmark** (NB6)
   - ~25 min on free T4

### 4. **Total time**: ~50 min

---

## Step-by-Step

### Option A: **Manual (Easy, just copy-paste)**
1. Open Colab: https://colab.research.google.com/github/<your-username>/Day22-Track3-DPO-Alignment-Lab/blob/main/colab/Lab22_DPO_T4.ipynb
2. Read: `COLAB_CELL_SNIPPETS.md` (this repo)
3. **Insert** the β override cell before Cell 54
4. **Run** cells 54 → 59 (wait ~18 min)
5. **Check** Cell 63 output for reward gap sign
6. **Re-run** cells 75-86 (evaluation)
7. **Run** cells 120-132 (benchmark)

### Option B: **Automatic (Using script)**
```bash
# On your laptop:
python3 scripts/fix_dpo_config.py --notebook colab/Lab22_DPO_T4.ipynb --beta 0.05

# Then upload the fixed notebook to Colab
# And run from there
```

---

## Expected Results (After Fix)

| Metric | Before | After (expected) |
|--------|--------|------------------|
| Reward gap | **-0.240** ❌ | **+0.05 ~ +0.20** ✅ |
| Judge win rate | 0/8 (ties) | 2-4/8 wins (improved) |
| Benchmark direction | (not run) | IFEval ↑, GSM8K ±, MMLU ±, AlpacaEval ↑ |

---

## Files Created for You

| File | Purpose |
|------|---------|
| `submission/REFLECTION.md` | ✅ Updated with actual results + analysis |
| `COLAB_FIX_GUIDE.md` | Detailed step-by-step guide |
| `COLAB_CELL_SNIPPETS.md` | Copy-paste ready cells for Colab |
| `scripts/fix_dpo_config.py` | Auto-fix script (optional) |

---

## Screenshots You Need

After running, capture these 4 PNGs to `submission/screenshots/`:

1. **`03-dpo-reward-curves.png`** ← Cell 61 output (check gap is positive!)
2. **`04-side-by-side-table.png`** ← Cell 81 output (new comparison)
3. **`04b-judge-summary.png`** ← Cell 86 output (win/loss/tie counts)
4. **`07-benchmark-comparison.png`** ← Cell 132 output (4-bar chart)

---

## The Plan 📅

```
Timeline (free T4):
- Now: Copy REFLECTION.md to submission/
- +5 min: Insert β override cell in Colab
- +25 min: Run DPO training (Cell 54-59)
- +3 min: Check reward gap (Cell 63)
- +3 min: Run evaluation (Cell 73-86)
- +25 min: Run benchmark (Cell 120-132)
- +5 min: Capture screenshots
- +2 min: Push to GitHub

Total: ~70 min (mostly waiting for training)
```

---

## Next Actions

1. **Read** `COLAB_FIX_GUIDE.md` for detailed steps
2. **Use** `COLAB_CELL_SNIPPETS.md` to copy-paste cells
3. **Run** on Colab (T4, free)
4. **Report back**:
   - Was reward gap positive or still negative?
   - Did judge win rate improve?
5. **Submit** to LMS once benchmark finishes

---

## Questions?

- ✅ REFLECTION.md: Detailed analysis of why DPO failed
- ✅ COLAB_FIX_GUIDE.md: Step-by-step with troubleshooting
- ✅ COLAB_CELL_SNIPPETS.md: Ready-to-paste code blocks

**Good luck! 🎯**
