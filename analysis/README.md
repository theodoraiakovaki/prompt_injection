# PromptShield analysis

Open [promptshield_analysis.ipynb](promptshield_analysis.ipynb) and run the setup cell, then the analysis cells from top to bottom. The notebook can run from the repository root or the `analysis/` folder. Select a Python kernel and restart it after package installation if requested.

## Inputs

The notebook reads the local JSON exports in `../cleaning/data/exact/` and `../cleaning/data/normalized/`. If these are missing, run the [cleaning notebook](../cleaning/promptshield_cleaning.ipynb) first. Cleaning rules and duplicate audits are documented in the [cleaning README](../cleaning/README.md).

Original raw files under `../data/` are optional: they are used only to display how many rows were removed. That field is unavailable when raw files are absent. Training and evaluation use the cleaned exports only, and data files are not modified.

## Contents

- Class proportions, prompt lengths, and dataset previews.
- Three logistic-regression baselines: TF-IDF, character count only, and TF-IDF plus standardized character count.
- The same model settings, seed, and threshold (0.5) for both cleaning variants.
- Matching validation/test rows, plus validation excluding normalized training overlap where necessary.
- Separate figures for class balance, lengths, and TF-IDF validation/test confusion matrices, ROC and precision-recall curves.
- Error analysis and test/train prompt pairs with similarity, predicted label, and actual label.
- Calculated findings and limitations.

`VARIANTS` selects the cleaning variants; `PRIMARY_VARIANT` selects detailed plots and examples. These results concern derived cleaned splits and should not be presented as original official-split results. Test results have already been inspected; keep tuning on validation and disclose this exposure. This notebook does not train a neural network.
