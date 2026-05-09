# Potential Issues Found in kaggle_lab22_dpo.py

## 🔴 **Critical Issues**

### 1. Missing `torch` import in `evaluate_models()` function (Line 346)
**Location**: Line 346: `with torch.no_grad():`
**Error**: `NameError: name 'torch' is not defined`
**Fix**: Add `import torch` at the top of the function
```python
def evaluate_models():
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import pandas as pd
    import torch  # ← ADD THIS
```

### 2. Memory leak in `evaluate_models()` → `generate()` function (Lines 332-351)
**Issue**: Model is loaded fresh for EACH adapter (SFT then DPO), causing memory buildup
**Current code**:
```python
def generate(adapter_path, prompts_list):
    model, tokenizer = FastLanguageModel.from_pretrained(...)  # Load model
    if (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(...)  # Load adapter
    # ... generate ...
    return outputs  # Model NOT unloaded!
```

**Result**: After SFT generation + DPO generation = 2 models in VRAM
**Risk**: OOM (Out of Memory) error on T4 16GB
**Fix**: Delete model after use
```python
def generate(adapter_path, prompts_list):
    m, tok = FastLanguageModel.from_pretrained(...)
    if (adapter_path / "adapter_config.json").exists():
        m = PeftModel.from_pretrained(m, str(adapter_path))
    FastLanguageModel.for_inference(m)
    
    outputs = []
    for prompt in prompts_list:
        inputs = tok(prompt, return_tensors="pt").to(m.device)
        with torch.no_grad():
            output = m.generate(**inputs, max_new_tokens=100, temperature=0.7)
        text = tok.decode(output[0], skip_special_tokens=True)
        outputs.append(text[len(prompt):].strip())
    
    # ← ADD CLEANUP:
    del m
    torch.cuda.empty_cache()
    
    return outputs
```

### 3. NB3 issue: Model passed to DPO but may be stale (Line 464)
**Issue**: In `main()`, `sft_model` is loaded in NB1, then passed to `run_dpo_training()`. Between training and passing, no guarantee model is in correct state.
**Current code** (Lines 443-453):
```python
if not SFT_PATH.exists():
    sft_model, tokenizer = run_sft_training()  # Fresh model, LoRA added
else:
    sft_model, tokenizer = FastLanguageModel.from_pretrained(...)  # Load base
    sft_model = PeftModel.from_pretrained(sft_model, str(SFT_PATH))  # Load LoRA
```

**Issue**: When loading from disk (else branch), LoRA adapter loaded but **NOT set for inference** (missing `FastLanguageModel.for_inference()`)
**Risk**: DPO training may behave differently
**Fix**: Add for_inference call:
```python
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
    # ← ADD THIS:
    FastLanguageModel.for_training(sft_model)  # Prepare for training
```

---

## 🟡 **Warnings (May cause issues)**

### 4. NB5 doesn't use LoRA adapter, loads fresh 16-bit model (Line 403)
**Location**: Lines 403-408
**Current code**:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(merged_path),  # Load merged model from disk
    max_seq_length=MAX_LEN,
)
```

**Issue**: Loading merged model is fine, but expects file to exist at `merged_path`
**Potential problem**: If NB5 runs before merge is complete, will fail
**Risk**: Low (should be ok if full pipeline runs sequentially)

### 5. No error handling or checkpoints for long operations
**Issue**: NB1 (15 min) and NB3 (18 min) don't save checkpoints. If interrupted:
- NB1 loss: Have to re-run everything
- NB3 loss: Have to re-train DPO from scratch

**Risk**: Medium (Kaggle can timeout, kernel can disconnect)
**Note**: `save_strategy="no"` is intentional to save disk space, but risky

### 6. Preference data loading in NB3 assumes parquet file (Line 258)
**Location**: Line 258
```python
pref_ds = load_dataset("parquet", data_files=str(PREF_PATH))["train"]
```

**Issue**: If NB2 hasn't run, file doesn't exist
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: '.../train.parquet'`
**Current guard**: Lines 456-460 check if file exists and run NB2 if needed ✓
**Status**: Already handled, OK

---

## ✅ **Already Fixed / OK**

- ✓ LoRA adapter setup (Line 141-150) - ADDED
- ✓ Dataset column names (Lines 157-167) - FIXED
- ✓ β=0.05 override (Line 100) - SET
- ✓ Preference data check (Line 456) - GUARDED
- ✓ SFT artifact check (Line 441) - GUARDED

---

## 🎯 **Recommended Fixes Before Running**

**MUST FIX**:
1. Add `import torch` in `evaluate_models()` function
2. Add `del m; torch.cuda.empty_cache()` in `generate()` function

**SHOULD FIX**:
3. Add `FastLanguageModel.for_training()` when loading SFT from disk

**NICE TO HAVE**:
4. Add error handling / try-except blocks
5. Add checkpoint saving for long operations

---

## 📊 **Memory Profile (Worst Case)**

| Step | Models in VRAM | Size | Total VRAM |
|------|---|---|---|
| NB1 | Base (4-bit) + LoRA | 3GB + 50MB | ~3.1 GB |
| NB2 | None | 0 | 0 GB |
| NB3 | Policy + Reference (frozen) | ~3.5GB + ~3.5GB | ~7 GB |
| NB4 SFT gen | Base + LoRA (SFT) | ~3.1 GB | ~3.1 GB |
| NB4 DPO gen | Base + LoRA (DPO) | ~3.1 GB | **~6.2 GB** ❌ |
| NB5 Merge | Merged model (fp16) | ~6 GB | **~6 GB** ❌ |

**T4 Free VRAM**: 14-15 GB
**Buffer**: ~2-3 GB left for OS + other processes
**Risk**: NB4 + NB5 close to limit

---

## 🚀 **Action Items**

Before running on Kaggle, apply these fixes:

```diff
# In evaluate_models() function, add import:
def evaluate_models():
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import pandas as pd
+   import torch

# In generate() function, add cleanup at end:
def generate(adapter_path, prompts_list):
    ... (existing code) ...
    outputs.append(text[len(prompt):].strip())

+   # Cleanup to avoid OOM
+   del m
+   torch.cuda.empty_cache()

    return outputs

# In main() function, add for_training when loading SFT:
else:
    ...
    sft_model = PeftModel.from_pretrained(sft_model, str(SFT_PATH))
+   FastLanguageModel.for_training(sft_model)
```

---

## Summary

- **Critical issues**: 1-2 (fixable in minutes)
- **Memory risks**: Moderate (OOM possible in NB4/NB5)
- **Recommended fix**: Add 3 lines of code
- **Risk level**: 🟡 MEDIUM (add fixes before running)
