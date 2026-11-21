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

## CNN model

Run the CNN notebooks in this order. Each contains saved results you can view
before running any cells.

1. [Train the CNNs](cnn/cnn_train.ipynb) to screen configurations, confirm the
   finalists, and save the refit models.
2. [Evaluate the CNNs](cnn/cnn_test.ipynb) to score validation and test data,
   inspect errors, and save the baseline test predictions.
3. [Run the prompt-length ablation](cnn/cnn_length_ablation.ipynb) to compare
   the text-only winner with raw and min-max scaled token counts.

### Environment

Use Python 3.14 and select your environment as the notebook kernel in VS Code
or Jupyter. The study used Python 3.14.4. For CPU evaluation or a smoke run,
activate a Python environment and install both requirements files from the
repository root.

    python -m pip install -r requirements.txt -r cnn/requirements-notebooks.txt

For full GPU training, install PyTorch for your GPU and driver in a separate
environment. Install the other dependencies from `requirements.txt` and all
dependencies from `cnn/requirements-notebooks.txt` in that environment.
Confirm that `torch.cuda.is_available()` is true in the selected kernel.
PyTorch uses `cuda:0` for both NVIDIA CUDA and AMD ROCm devices.

### Run settings and data

The first two notebooks default to `RUN_NAME = "cnn_study_v2_full_01"`.
If you choose a different name, use it in all three notebooks. The training
notebook uses `DEVICE_REQUEST = "cuda:0"`, and the evaluation notebook can
run on CPU. The ablation uses the baseline run's GPU and software versions.

For a quick CPU check, set `SMOKE_MODE = True` and `DEVICE_REQUEST = "cpu"`
in the first two notebooks, and set `RUN_TEST = False` in the evaluation
notebook. The length ablation requires a completed full study with test
predictions.

The CNN uses the published raw splits in `data/` and the matching normalized
splits in `data/normalized/`. The saved TF-IDF comparison also reads the
exact-deduplicated train and test splits in `cleaning/data/exact/`. The
normalization in `data/normalized/` uses lowercase copies of published raw
prompts and preserves the study's row order. See the
[data provenance and license](data/README.md).

The notebooks generate checkpoints and reports under `cnn/artifacts/`.
Run training in your clone to create a manifest with your local data paths.
