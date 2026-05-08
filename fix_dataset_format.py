"""
Debug script: Check actual dataset structure and fix format function
"""

import subprocess
import sys

# Install deps
packages = ["datasets", "pyarrow", "pandas"]
for pkg in packages:
    print(f"Installing {pkg}...", end=" ", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✓")

print("\n" + "="*60)
print("Checking Vietnamese Alpaca dataset structure")
print("="*60 + "\n")

from datasets import load_dataset

# Load first sample
ds = load_dataset("5CD-AI/Vietnamese-alpaca-gpt4-gg-translated", split="train[:1]")

print(f"Dataset columns: {ds.column_names}")
print(f"Dataset size: {len(ds)}")

print("\n🔹 First sample:")
example = ds[0]
for key, value in example.items():
    print(f"\n[{key}]:")
    print(f"  Type: {type(value)}")
    print(f"  Value: {str(value)[:100]}...")

# Generate correct format function
print("\n" + "="*60)
print("Correct format function:")
print("="*60 + "\n")

cols = set(example.keys())
print(f"Available columns: {cols}")

# Build format function based on actual columns
if 'instruction' in cols and 'output' in cols:
    format_code = '''def format_chat(example):
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    if input_text:
        text = f"### Instruction:\\n{instruction}\\n### Input:\\n{input_text}\\n### Output:\\n{output}"
    else:
        text = f"### Instruction:\\n{instruction}\\n### Output:\\n{output}"

    return {"text": text}'''

elif 'prompt' in cols and 'response' in cols:
    format_code = '''def format_chat(example):
    prompt = example.get('prompt', '')
    response = example.get('response', '')

    text = f"### Prompt:\\n{prompt}\\n### Response:\\n{response}"

    return {"text": text}'''

elif 'input' in cols and 'output' in cols:
    format_code = '''def format_chat(example):
    input_text = example.get('input', '')
    output = example.get('output', '')

    text = f"### Input:\\n{input_text}\\n### Output:\\n{output}"

    return {"text": text}'''

else:
    # Fallback: use first 2 columns
    cols_list = list(cols)
    format_code = f'''def format_chat(example):
    col1 = example.get('{cols_list[0]}', '')
    col2 = example.get('{cols_list[1]}', '') if len({cols_list}) > 1 else ''

    text = f"{{col1}}\\n\\n{{col2}}"

    return {{"text": text}}'''

print(format_code)
print("\n" + "="*60)
print("Copy the function above into your script!")
print("="*60)
