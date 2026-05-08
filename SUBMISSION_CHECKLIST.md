# Submission Checklist — Lab 22

**Due**: 23:59 next day | **Late penalty**: -10% per day | **Regrade**: within 1 week

---

## ✅ Core Submission Items (100 pts)

### Notebooks (30 pts)
- [ ] `01_sft_mini.ipynb` — output cells preserved, SFT loss curve shows monotonic decrease
- [ ] `02_preference_data.ipynb` — parquet written, 3 examples printed
- [ ] `03_dpo_train.ipynb` — adapter saves, reward curves plotted
- [ ] `04_compare_and_eval.ipynb` — side-by-side table with ≥8 prompts, win/loss reported
- [ ] `05_merge_deploy_gguf.ipynb` — GGUF Q4_K_M created, smoke test shows VN output

**OR**: Single `colab/Lab22_DPO_T4.ipynb` with all cells executed

### Screenshots (6 required → `submission/screenshots/`)

#### Critical Screenshots
- [ ] `01-setup-gpu.png` — nvidia-smi or torch.cuda output (T4 16GB visible)
- [ ] `02-sft-loss.png` — SFT loss curve (monotonic ↓ over 1 epoch)
- [ ] `03-dpo-reward-curves.png` — **BOTH chosen & rejected curves** + gap (check sign!)
  - ✅ Good: gap goes UP (chosen ↑ or rejected ↓)
  - ❌ Bad: gap goes DOWN (our current: -0.240)
- [ ] `04-side-by-side-table.png` — ≥8 prompts × 2 outputs (SFT | DPO) × category labels
- [ ] `05-judge-output.png` — Claude/GPT judge verdicts (or manual rubric filled)
- [ ] `06-gguf-smoke.png` — llama-cpp-python loads GGUF, generates coherent VN

#### Quantitative Screenshot
- [ ] `07-benchmark-comparison.png` — 4-bar chart (IFEval, GSM8K, MMLU, AlpacaEval) with Δ annotated

### REFLECTION.md (25 pts)

- [ ] §1 Setup — all fields filled (GPU, CUDA, model, dataset, cost)
- [ ] §2 Results table — final loss, VRAM, reward gap, output length
- [ ] §3 Reward curves (≥100 words) — interpret BOTH chosen & rejected
  - Explain: Did chosen go UP? Or gap grow because rejected dropped faster?
  - Reference deck §3.4 (likelihood displacement)
  - **Current**: "negative gap = DPO learned backwards"
- [ ] §4 Qualitative comparison — ≥8 examples with winner column + summary
  - **Current**: "0/8 wins (all ties)" ← discuss why
- [ ] §5 β trade-off — either:
  - [ ] Ran β-sweep {0.05, 0.1, 0.5} with results table, OR
  - [ ] 3-sentence hypothesis if you didn't sweep
- [ ] §6 Personal reflection (≥150 words) — walk through ONE decision
  - What was alternative? Why chose this one? Did it confirm/surprise? Next time?
  - **Suggested**: "Why choose UltraFeedback English? → Negative gap → next: switch to VN data"
- [ ] §7 Benchmark interpretation (≥150 words) — which went up/down, alignment tax, final verdict
  - **Current**: "Not run yet" ← run & fill in

### Code Quality (5 pts)
- [ ] `make verify` exits 0
  - Checks: adapters/sft-mini/, adapters/dpo/, data/pref/*.parquet, gguf/*.gguf exist
  - Checks: all output PNG/JSON present
  - Checks: REFLECTION.md has required sections

### Reproducibility (5 pts)
- [ ] Code runs clean from `setup-laptop.sh` + `make pipeline` (or Colab Run-all)
- [ ] No hardcoded paths, all use environment variables
- [ ] No broken imports

---

## 🎯 Current Status

| Item | Status | Action |
|------|--------|--------|
| SFT training | ✅ Done | — |
| Pref data prep | ✅ Done | — |
| DPO training | ⚠️ Negative gap | **Re-run with β=0.05** |
| Evaluation | ⚠️ All ties | Re-run after DPO fix |
| Merge & GGUF | ✅ Done | — |
| Benchmark | ❌ Not run | **Run NB6** |
| REFLECTION.md | ✅ Written | ✅ Updated! |
| Screenshots | ⚠️ Partial | **Capture 4 more after fix** |
| make verify | ❌ Will fail | Fix artifacts + re-run |

---

## 🎨 Optional Rigor Add-ons (+20 pts max, pick any)

- [ ] **β-sweep** (+6): Run NB3 with {0.05, 0.1, 0.5}, plot gap vs β, 100+ word interpretation
- [ ] **HuggingFace push** (+5): Push DPO adapter with model card
- [ ] **GGUF release** (+3): Push GGUF with Q4_K_M + Q5_K_M quantizations
- [ ] **MMLU full** (+3): Run NB6 with LIMIT_MMLU=14000 (full)
- [ ] **W&B link** (+2): Public wandb run URL with curves
- [ ] **Cross-judge** (+4): Judge with gpt-4o-mini AND claude-haiku, report disagreement %
- [ ] **Bonus challenge** (ungraded): Creative exploration in `bonus/` folder

---

## 📦 Submission Format

### Option A — Lightweight (default)
```
YOUR_REPO/
├── colab/Lab22_DPO_T4.ipynb (executed, output cells preserved)
├── submission/
│   ├── REFLECTION.md (all 7 sections filled)
│   └── screenshots/
│       ├── 01-setup-gpu.png
│       ├── 02-sft-loss.png
│       ├── 03-dpo-reward-curves.png
│       ├── 04-side-by-side-table.png
│       ├── 05-judge-output.png
│       ├── 06-gguf-smoke.png
│       └── 07-benchmark-comparison.png
├── adapters/ (sft-mini, dpo)
├── data/eval/benchmark_results.json
└── gguf/*.gguf
```

### Option B — Professional (+5 bonus)
All of Option A +
```
├── adapters/dpo/ pushed to HuggingFace
├── README.md links to HF model
└── model card (base, dataset, hyperparams, eval results)
```

### Option C — Code-only (no weights)
Option A but skip uploading large artifacts (adapters/, gguf/) — still eligible for full core points.

---

## 🚀 Next Steps (Do This Now)

1. **Read** `README_QUICK_FIX.md` (1 min)
2. **Read** `COLAB_FIX_GUIDE.md` (5 min)
3. **Open** Colab notebook
4. **Insert** β override cell (2 min)
5. **Run** cells 54 → 59 (18 min ⏳)
6. **Check** reward gap sign in cell 63
7. **Re-run** cells 75-86 (3 min)
8. **Run** cells 120-132 (25 min ⏳)
9. **Capture** 4 new screenshots
10. **Update** REFLECTION.md §2 & §7 with new numbers
11. **Run** `make verify` locally
12. **Push** to GitHub (repo must stay public)
13. **Submit** repo URL to LMS

**Total time**: ~60 min (mostly waiting for training)

---

## 🔍 Grading Rubric Mapping

Each requirement below maps to rubric.md section:

| Rubric | Requirement | Current | After Fix |
|--------|-------------|---------|-----------|
| 1 | SFT adapter r=16 | ✅ | ✅ |
| 2 | SFT loss ↓ | ✅ | ✅ |
| 3 | SFT generation | ✅ | ✅ |
| 4 | Pref data saved | ✅ | ✅ |
| 5 | Pref examples | ✅ | ✅ |
| 6 | DPO adapter | ✅ | ✅ |
| 7 | **Reward gap plot** | ❌ (-0.240) | ⏳ (will test) |
| 8 | **Both curves plotted** | ⚠️ (only gap) | ✅ (add both) |
| 9 | Side-by-side ≥8 | ✅ | ✅ (re-run) |
| 10 | Win/loss reported | ✅ | ✅ (update) |
| 11 | GGUF < 5 GB | ✅ | ✅ |
| 12 | GGUF smoke | ✅ | ✅ |
| 13 | Benchmark JSON | ❌ | ⏳ (will create) |
| 14 | Benchmark plot | ❌ | ⏳ (will create) |
| 15 | Code reproducible | ⚠️ | ✅ (after fix) |
| 16 | REFLECTION complete | ⚠️ | ✅ |
| 17 | Reward curve interp | ⚠️ | ✅ |
| 18 | Benchmark interp | ❌ | ⏳ (will add) |
| 19 | Make verify passes | ❌ | ⏳ (after all) |

**Current est.**: ~65/100 → **After fix est.**: ~95/100 (if β fix works)

---

## 💡 Tips

1. **Keep notebook output cells** — don't clear them
2. **Crop screenshots tight** — no wallpaper/distractions
3. **Never commit API keys** — recrop if visible
4. **Test locally** (`make verify`) before submitting
5. **Keep repo public** — private = 0 points
6. **If β=0.05 still fails** → try β=0.02 or mention "data quality issue" in REFLECTION §3
7. **If you run out of time** → submit what you have + explain in REFLECTION what's pending

---

## 📞 Help

- Error in DPO training? → Check Cell 50-58 (model setup)
- Reward gap still wrong? → Mention in REFLECTION, try β=0.02
- Benchmark times out? → Reduce LIMIT_* values
- Screenshots missing? → Capture from output cells

Good luck! 🎯
