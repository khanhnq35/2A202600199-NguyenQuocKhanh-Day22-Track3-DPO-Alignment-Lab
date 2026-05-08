#!/usr/bin/env python3
"""
Auto-fix script: Replace β=0.1 with β=0.05 in Colab notebook
Usage: python scripts/fix_dpo_config.py --notebook colab/Lab22_DPO_T4.ipynb
"""

import json
import argparse
import re
from pathlib import Path


def fix_dpo_config(notebook_path, beta_value=0.05):
    """
    Find and replace the DPOConfig cell with updated β value.
    """
    notebook_path = Path(notebook_path)

    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False

    print(f"📖 Loading notebook: {notebook_path}")

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Find the DPOConfig cell
    found = False
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue

        source = ''.join(cell.get('source', []))

        # Look for the cell with DPOConfig and beta=
        if 'DPOConfig' in source and 'beta=' in source:
            print(f"✅ Found DPOConfig cell at index {i}")

            # Replace beta value
            old_source = source

            # Match: beta=0.1 or beta = 0.1
            new_source = re.sub(
                r'beta\s*=\s*0\.\d+',
                f'beta={beta_value}',
                source
            )

            # Also update any comments about beta
            if "β =" in new_source or "β=" in new_source:
                new_source = re.sub(
                    r'(β\s*=\s*)0\.\d+',
                    f'\\1{beta_value}',
                    new_source
                )

            # Update the cell
            cell['source'] = new_source.splitlines(keepends=True)

            # Show diff
            if old_source != new_source:
                print(f"\n🔄 Changed β from 0.1x to {beta_value}:")
                print(f"   OLD: beta = 0.1x")
                print(f"   NEW: beta = {beta_value}")
                found = True
            else:
                print("⚠️  No change needed (already at target β?)")
                found = False

            break

    if not found:
        print("❌ Could not find DPOConfig cell with beta parameter")
        return False

    # Save the modified notebook
    output_path = notebook_path.parent / f"{notebook_path.stem}_fixed.ipynb"
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\n✅ Fixed notebook saved to: {output_path}")
    print("\nNext steps:")
    print(f"1. Open: {output_path}")
    print("2. Run cells 54 → 65 (DPO training with β=0.05)")
    print("3. Check reward gap in cell 63 output")
    print("4. Run cells 73 → 86 (new comparison)")
    print("5. Run cells 120+ (benchmark)")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix DPO config β value in notebook')
    parser.add_argument('--notebook', default='colab/Lab22_DPO_T4.ipynb',
                       help='Path to notebook file')
    parser.add_argument('--beta', type=float, default=0.05,
                       help='New β value (default: 0.05)')

    args = parser.parse_args()

    success = fix_dpo_config(args.notebook, args.beta)
    exit(0 if success else 1)
