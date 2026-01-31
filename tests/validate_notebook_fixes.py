#!/usr/bin/env python3
"""
Validate that the notebook fixes are correct.
"""
import json
from pathlib import Path

def validate_notebook():
    """Check that all critical cells are correctly configured."""

    notebook_path = Path("notebooks/02_model_fitting.ipynb")

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    print("=" * 70)
    print("NOTEBOOK VALIDATION")
    print("=" * 70)

    issues = []

    # Check Cell 5 (ModelConfig)
    print("\n[Cell 5] ModelConfig:")
    cell_5 = ''.join(nb['cells'][5]['source'])
    if 'separate_kicking_effect=True' in cell_5:
        print("  ✓ separate_kicking_effect explicitly set")
    else:
        issues.append("Cell 5: separate_kicking_effect not explicitly set")
        print("  ✗ separate_kicking_effect not found")

    if 'include_defense=True' in cell_5:
        print("  ✓ include_defense explicitly set")
    else:
        print("  ⚠ include_defense not explicitly set (may use default)")

    # Check Cell 14 (show_rankings function)
    print("\n[Cell 14] show_rankings function:")
    cell_14 = ''.join(nb['cells'][14]['source'])

    if 'def show_rankings(score_type, top_n=20, min_scores=' in cell_14:
        print("  ✓ Function signature has min_scores parameter")
    else:
        issues.append("Cell 14: min_scores parameter missing")
        print("  ✗ min_scores parameter missing")

    if 'df=df' in cell_14 and 'min_scores=min_scores' in cell_14:
        print("  ✓ Function passes df and min_scores to get_player_rankings")
    else:
        issues.append("Cell 14: Not passing df or min_scores to get_player_rankings")
        print("  ✗ Not passing df/min_scores to get_player_rankings")

    if 'min_scores=10' in cell_14:
        print("  ✓ Tries minimum set to 10")
    else:
        print("  ⚠ Tries minimum may not be 10")

    if 'min_scores=20' in cell_14:
        print("  ✓ Kicks minimum set to 20")
    else:
        print("  ⚠ Kicks minimum may not be 20")

    # Check Cell 15 (Visualization)
    print("\n[Cell 15] Visualization:")
    cell_15 = ''.join(nb['cells'][15]['source'])

    # Check for string keys (correct)
    if "'tries':" in cell_15 or '"tries":' in cell_15:
        print("  ✓ Using string keys in threshold dictionary")
    else:
        issues.append("Cell 15: Not using string keys (syntax error)")
        print("  ✗ Missing string keys - will cause NameError")

    if 'df=df' in cell_15 and 'min_scores=min_threshold' in cell_15:
        print("  ✓ Passing df and min_scores to get_player_rankings")
    else:
        issues.append("Cell 15: Not passing df or min_scores")
        print("  ✗ Not passing df/min_scores")

    if "'total_scores' in rankings.columns" in cell_15:
        print("  ✓ Checking for total_scores column")
    else:
        print("  ⚠ May not be displaying score counts")

    # Check Cell 18 (Best Kickers)
    print("\n[Cell 18] Best Kickers:")
    cell_18 = ''.join(nb['cells'][18]['source'])

    if cell_18.count('df=df') >= 2:
        print("  ✓ Both penalty and conversion rankings use df parameter")
    else:
        issues.append("Cell 18: Not passing df to both rankings")
        print("  ✗ Missing df parameter in one or both rankings")

    if cell_18.count('min_scores=20') >= 2:
        print("  ✓ Both penalty and conversion rankings use min_scores=20")
    else:
        issues.append("Cell 18: Not using min_scores=20 for both")
        print("  ✗ Missing min_scores=20 in one or both rankings")

    # Summary
    print("\n" + "=" * 70)
    if len(issues) == 0:
        print("✅ ALL CHECKS PASSED - Notebook is correctly configured!")
    else:
        print(f"❌ FOUND {len(issues)} ISSUE(S):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    print("=" * 70)

    return len(issues) == 0

if __name__ == "__main__":
    success = validate_notebook()
    exit(0 if success else 1)
