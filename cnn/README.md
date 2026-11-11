# CNN prompt-injection detector

## View the completed full run

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
[`artifacts/gpu_focused_v1/`](artifacts/gpu_focused_v1/):

- [leaderboard.csv](artifacts/gpu_focused_v1/leaderboard.csv): all eight configurations.
- [manifest.json](artifacts/gpu_focused_v1/manifest.json): configurations, seeds, data hashes, and fold assignments.
- [selected.json](artifacts/gpu_focused_v1/selected.json): selected settings and final training history.
- [validation_report.json](artifacts/gpu_focused_v1/analysis/validation_report.json): reference comparisons and all decision profiles.
- Per-fold `result.json` files: learning histories and held-out metrics, available without retraining.
- Validation prediction files and the frozen evaluation record: scores, row indices, and chosen thresholds.

The selected [model checkpoint](artifacts/gpu_focused_v1/selected.pt)
includes its trained weights, vocabulary, and settings. To load it on CPU without
training, use:

```python
from cnn.cnn_experiments import load_checkpoint

model, vocabulary, config, metadata = load_checkpoint(
    "cnn/artifacts/gpu_focused_v1/selected.pt", device="cpu"
)
```

The frozen record retains original-machine model paths for provenance; saved
notebook outputs and the selected checkpoint can be viewed or loaded independently.
Per-fold model weights, smoke runs, environment folders, and local GPU diagnostics
are not part of the published outcome. Use a new `RUN_NAME` for a new experiment.

## Run a new experiment

Continue from [the baseline analysis](../promptshield_analysis.ipynb) with these two notebooks. Both use numbered
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
uv pip install --python .venv/Scripts/python.exe -r cnn/requirements-notebooks.txt
# For the existing AMD training environment instead:
uv pip install --python gpu_test/.venv/Scripts/python.exe -r cnn/requirements-notebooks.txt
```

These commands do not request a different PyTorch build. Select that interpreter
as the notebook kernel; do not install the root CPU requirements over the ROCm
environment. CPU execution is also supported. The notebooks print and record the
actual device. GPU trials run sequentially.

New results are written under ignored `cnn/artifacts/`; the selected
full run's reports and checkpoint are explicitly versioned. Completed matching
folds resume; changes to source, data, settings, or runtime require a new run folder.
Original analysis files and checkpoints are preserved. Threshold-selected validation
metrics are development results. The previously inspected test split is a known
benchmark; an untouched dataset is needed for an unbiased final assessment.

Run these checks from the repository root in the environment with notebook dependencies:

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v cnn.test_cnn_experiments
.\.venv\Scripts\python.exe -B cnn/verify_cnn_notebooks.py
```

The second command executes smoke copies of both notebooks and saves their outputs
under `cnn/artifacts/_verification/`; source notebooks stay unexecuted.


## Folder layout and historical records

The notebooks and shared code live in this folder. The original shared dataset
remains in `../data/`; the separate cleaning workflow remains in `../cleaning/`.
The retired root `cnn.py` is no longer required by the notebooks or tests.

Original manifests, checkpoints, predictions, and frozen records are moved without
changing their contents. Their source hashes and machine paths describe the original
experiment. Loading the completed search or selected checkpoint does not retrain it.
If you rerun validation with the reorganized module, the fresh report and frozen
thresholds go into `analysis_<source hash>/`, preserving the published `analysis/`
records. Use the resulting `FROZEN_PATH` for subsequent optional test evaluation.

Local legacy scripts and earlier checkpoints are archived under ignored
`artifacts/legacy/`; they are not required to view the published results.
