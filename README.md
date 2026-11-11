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

All CNN notebooks, shared model code, tests, and published experiment results live in
**[cnn/](cnn/README.md)**. The shared dataset remains in `data/`.

- **[Tuning results](cnn/cnn_tuning.ipynb)**: eight CNN configurations evaluated across five folds.
- **[Validation analysis](cnn/cnn_analysis.ipynb)**: reference comparisons, decision thresholds, and error review.
- **[Numerical leaderboard](cnn/artifacts/gpu_focused_v1/leaderboard.csv)**: results for all 40 fold runs.

The selected `cnn06` model achieved **98.03% mean fold macro F1** and **97.80%
validation injection F1 at threshold 0.5**. Its checkpoint and saved notebook outputs
are included, so the team can view the outcome without rerunning training.
Test evaluation of the selected checkpoint is pending.

See [cnn/README.md](cnn/README.md) for settings, dependency installation, model
loading, and verification. The separate data-cleaning workflow is in
[cleaning/](cleaning/README.md).
