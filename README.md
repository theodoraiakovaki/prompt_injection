# Prompt Injection Detection and Localization: A Cross-Lingual Deep Learning Approach

## Project Description

This project aims to investigate prompt injection detection using deep learning techniques. Using the [PromptShield dataset](https://huggingface.co/datasets/hendzh/PromptShield), we will develop and evaluate a model that classifies prompts as benign or malicious.

The main objective is to investigate the effectiveness of supervised learning for prompt injection detection, with unsupervised anomaly detection considered as a potential alternative for identifying malicious prompts without relying on labeled training examples.

As an extension, we will explore prompt injection localization, aiming to identify the exact text spans responsible for an attack. This may involve comparing unsupervised anomaly detection at the text-segment level with supervised token classification, depending on the availability of suitable annotations.

Finally, we will investigate the potential for cross-lingual generalization by evaluating whether a model trained primarily on English data can detect prompt injection attacks in Greek.

## Main Objective

- Develop and evaluate a prompt injection detector using supervised learning.

## Potential Extensions

- Compare supervised detection with unsupervised anomaly detection.
- Investigate localization of prompt injection attacks at the text-span or token level.
- Evaluate cross-lingual generalization from English to Greek.

## Dataset

[PromptShield — Hugging Face](https://huggingface.co/datasets/hendzh/PromptShield)

## Project Status

Project proposal — implementation and experimental evaluation are pending.
