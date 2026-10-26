# Prompt injection detection

Explore the [PromptShield dataset](https://huggingface.co/datasets/hendzh/PromptShield) and build a first prompt injection detector in **[promptshield_analysis.ipynb](promptshield_analysis.ipynb)**.

## Use the notebook

1. Open the notebook in VS Code with the Jupyter extension, or in JupyterLab.
2. Select a Python kernel (the existing `.venv` environment is ready to use).
3. Run the cells from top to bottom. The first code cell installs the analysis packages.

The notebook contains local data loading, data preview, class balance, length analysis, duplicate and split-overlap checks, plots, TF-IDF/logistic regression baseline, validation metrics, and findings. It is self-contained and does not import project scripts.

The notebook reads `train.json`, `validation.json`, and `test.json` directly from the existing `data/` folder. Run it from the repository root, or set `DATA_DIR` to your data folder. Data loading requires no internet access. Results, tables, and plots appear directly in notebook cells. Test data is inspected for the audit; test model evaluation is reserved for the final experiment.

The project aims to compare supervised and unsupervised detection. Unsupervised methods, attack localization, and Greek evaluation are potential next steps.
