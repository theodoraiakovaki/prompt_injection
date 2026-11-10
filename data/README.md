# PromptShield data

These JSON splits come from [hendzh/PromptShield](https://huggingface.co/datasets/hendzh/PromptShield), revision `a5234cb1f5cdb256600cab64b8c961195b5e8404`. Train and validation are unmodified. Test has the two redactions described below.

The upstream dataset card lists the license as **Apache 2.0**; the license text is included in [LICENSE](LICENSE). See the [upstream dataset card](https://huggingface.co/datasets/hendzh/PromptShield/blob/a5234cb1f5cdb256600cab64b8c961195b5e8404/README.md) for provenance and citation information.

| File | Rows |
| --- | ---: |
| `train.json` | 18,909 |
| `validation.json` | 1,000 |
| `test.json` | 23,516 |

`prompt` is the input text. `label` is `0` for benign and `1` for injection. `manifest.json` records source URLs and SHA-256 hashes of the published files. Attack instructions inside the prompts are dataset content.

## Publication redactions

GitHub push protection detected one OpenAI-key-shaped string and one Hugging-Face-token-shaped string in the upstream test data. Both were replaced with `REDACTED_...` placeholders of the same character length. No token validity checks were performed. Labels, row counts, prompt character lengths, and whitespace word counts are unchanged; the two string values and test-file hash differ from upstream. Train and validation model results are unaffected, and no test model scoring has been run.

The manifest includes both the original and published test-file hashes and redaction counts. The original test file is retained locally in ignored `data/originals/test.json`; it is not published. For exact upstream reproduction, obtain the original test split from the pinned source URL in the manifest.

The accompanying notebook reads these files locally. Model training uses train, development evaluation uses validation, and test model evaluation remains reserved.

## Citation

Dennis Jacob, Hend Alzahrani, Zhanhao Hu, Basel Alomair, and David Wagner. *PromptShield: Deployable Detection for Prompt Injection Attacks* (2025). [arXiv:2501.15145](https://arxiv.org/abs/2501.15145).
