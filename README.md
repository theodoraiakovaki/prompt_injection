# Project Overview

This project investigates prompt injection detection using deep learning and machine learning techniques. Using the PromptShield dataset, we aim to distinguish malicious prompts from benign ones and evaluate the effectiveness of different detection approaches.

## Main Objectives

- **Develop a prompt injection detection system** that classifies prompts as benign or malicious.
- **Compare supervised and unsupervised approaches** to investigate their effectiveness in detecting prompt injection attacks.
- **Evaluate model performance** using appropriate metrics, such as precision, recall, F1-score, and confusion matrices.
- **Investigate the limitations of each approach**, particularly the challenges of detecting malicious prompts when labeled training data are unavailable.
- **Explore a potential extension**, either identifying the specific text spans responsible for prompt injection attacks (localization) or evaluating cross-lingual generalization by testing detection on Greek prompts.

## Dataset

We plan to use the [PromptShield dataset](https://huggingface.co/datasets/hendzh/PromptShield), which contains prompts labeled as benign or malicious. The dataset will be examined and preprocessed before being used for model development and evaluation.

## Methodology

The project will investigate two main approaches:

- **Supervised learning:** Train a classification model using labeled examples to distinguish benign prompts from prompt injection attacks.
- **Unsupervised learning:** Explore anomaly detection or clustering techniques to investigate whether potentially malicious prompts can be identified without using labels during training.

Both approaches will be evaluated using a consistent experimental setup, with results compared to understand their strengths and limitations.

## Potential Extensions

Depending on feasibility and data availability, we may extend the project in one of two directions:

- **Prompt injection localization:** Identify the specific words or text spans that contain injection attempts, rather than only classifying the entire prompt.
- **Cross-lingual generalization:** Investigate whether a detector trained on English prompts can identify prompt injection attacks in Greek.

## Expected Outcomes

The project aims to provide an experimental comparison of supervised and unsupervised approaches to prompt injection detection, highlighting their performance, limitations, and suitability for scenarios with limited labeled data. The optional extension may provide further insight into either attack localization or cross-lingual robustness.

## CNN experiments

### View the completed full run

Open **[cnn_tuning.ipynb](cnn_tuning.ipynb)** and then
**[cnn_analysis.ipynb](cnn_analysis.ipynb)** directly on GitHub. Both notebooks
include their executed tables and plots; viewing the outcome requires no Python
environment, GPU, or rerun. These saved outputs are from the full run, not smoke mode.

The `gpu_focused_v1` experiment completed all **40 fold runs**: eight configurations
across five grouped, stratified training folds. Its selected configuration is
**cnn06**: one convolution layer, 256 filters, kernel size 5, 64-dimensional embeddings,
dropout 0.3, learning rate 0.001, and weight decay 0.0001. It was retrained on all
18,909 training prompts for seven epochs.

| Result | Value |
| --- | ---: |
| Mean held-out fold macro F1 | 98.03% |
| Fold macro-F1 standard deviation | 0.26 percentage points |
| Validation injection F1 at threshold 0.5 | 97.80% |
| Validation injection precision / recall at threshold 0.5 | 98.59% / 97.02% |
| Validation false positives / missed injections at threshold 0.5 | 7 / 15 |

Validation contains 1,000 prompts; threshold selection excludes 11 normalized
training overlaps. Threshold-selected validation results are development results.
**Test evaluation of this selected checkpoint is pending.**

Published numerical results are under
[`artifacts/cnn_experiments/gpu_focused_v1/`](artifacts/cnn_experiments/gpu_focused_v1/):

- [leaderboard.csv](artifacts/cnn_experiments/gpu_focused_v1/leaderboard.csv): all eight configurations.
- [manifest.json](artifacts/cnn_experiments/gpu_focused_v1/manifest.json): configurations, seeds, data hashes, and fold assignments.
- [selected.json](artifacts/cnn_experiments/gpu_focused_v1/selected.json): selected settings and final training history.
- [validation_report.json](artifacts/cnn_experiments/gpu_focused_v1/analysis/validation_report.json): reference comparisons and all decision profiles.
- Per-fold `result.json` files: learning histories and held-out metrics, available without retraining.
- Validation prediction files and the frozen evaluation record: scores, row indices, and chosen thresholds.

The selected [model checkpoint](artifacts/cnn_experiments/gpu_focused_v1/selected.pt)
includes its trained weights, vocabulary, and settings. To load it on CPU without
training, use:

```python
from cnn_experiments import load_checkpoint

model, vocabulary, config, metadata = load_checkpoint(
    "artifacts/cnn_experiments/gpu_focused_v1/selected.pt", device="cpu"
)
```

The frozen record retains original-machine model paths for provenance; saved
notebook outputs and the selected checkpoint can be viewed or loaded independently.
Per-fold model weights, smoke runs, environment folders, and local GPU diagnostics
are not part of the published outcome. Use a new `RUN_NAME` for a new experiment.

### Run a new experiment

Continue from `promptshield_analysis.ipynb` with these two notebooks. Both use numbered
steps with explanations and can be worked through in parts:

1. **[cnn_tuning.ipynb](cnn_tuning.ipynb)**: inspect training data and grouped folds,
   compare eight CNN configurations with five-fold cross-validation, review the
   leaderboard, and then separately retrain the selected configuration.
2. **[cnn_analysis.ipynb](cnn_analysis.ipynb)**: load saved results, inspect learning
   curves and validation errors, compare with the existing CNN and TF-IDF baseline,
   freeze decision thresholds, and optionally score the known test benchmark.

Start with `SMOKE_MODE = True` in both notebooks. This runs a small workflow check;
it does not establish a winning model. Set it to `False` in both notebooks for the
full 40-fold-run comparison. The final retraining is a separate notebook step.
Test scoring defaults to `RUN_TEST = False` and rejects smoke checkpoints.

Keep the current tokenizer, vocabulary limit, and sequence length fixed. Within
each training fold, early stopping uses a separate grouped subset and vocabulary
fitting uses only weight-training examples. Official validation and test are not
opened by the tuning notebook. Shared functions live in `cnn_experiments.py`.

Install notebook-only dependencies into the environment you intend to use:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-notebooks.txt
# For the existing AMD training environment instead:
uv pip install --python gpu_test/.venv/Scripts/python.exe -r requirements-notebooks.txt
```

These commands do not request a different PyTorch build. Select that interpreter
as the notebook kernel; do not install the root CPU requirements over the ROCm
environment. CPU execution is also supported. The notebooks print and record the
actual device. GPU trials run sequentially.

New results are written under ignored `artifacts/cnn_experiments/`; the selected
full run's reports and checkpoint are explicitly versioned. Completed matching
folds resume; changes to source, data, settings, or runtime require a new run folder.
Original analysis files and checkpoints are preserved. Threshold-selected validation
metrics are development results. The previously inspected test split is a known
benchmark; an untouched dataset is needed for an unbiased final assessment.

Verification in the environment with notebook dependencies:

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_cnn_experiments
.\.venv\Scripts\python.exe -B verify_cnn_notebooks.py
```

The second command executes smoke copies of both notebooks and saves their outputs
under `artifacts/cnn_experiments/_verification/`; source notebooks stay unexecuted.
