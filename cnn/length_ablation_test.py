"""Refit the CNN length variants and compare them on the saved test benchmark."""

from dataclasses import asdict
import json
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

from . import cnn_experiments as exp
from . import cnn_study as study
from .length_ablation import _LengthCNN, _loader, _write_result
from .length_features import length_counts, original_text


def _encode(rows, vocabulary, config):
    return (
        study.encode_rows(rows, vocabulary, config.preprocessing),
        torch.tensor([row["label"] for row in rows], dtype=torch.float32),
    )


def _features(counts, state):
    return (counts.reshape(-1, 1) * state["scale"] + state["offset"]).astype(np.float32)


def _save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def refit_length_models(
    output_dir, config, rows, epochs_by_variant, *, dataset, baseline_signature,
    seed, micro_batch_size, device, progress=print,
):
    """Fit and save both variants with epoch counts already fixed by cross-validation."""
    device = exp.resolve_device(device)
    if set(epochs_by_variant) != {"raw_tokens", "minmax_tokens"}:
        raise ValueError("Specify the epoch count for both length variants.")
    if any(epochs < 1 for epochs in epochs_by_variant.values()):
        raise ValueError("Each refit requires at least one epoch.")
    if config.length_feature != "none" or micro_batch_size < 1 or config.batch_size % micro_batch_size:
        raise ValueError("Expected the text-only CNN settings and a matching microbatch size.")

    counts = length_counts(rows, "log_tokens")
    scaler = MinMaxScaler(clip=False).fit(counts.reshape(-1, 1))
    feature_states = {
        "raw_tokens": {"scale": 1.0, "offset": 0.0},
        "minmax_tokens": {
            "scale": float(scaler.scale_[0]), "offset": float(scaler.min_[0]),
            "minimum": int(scaler.data_min_[0]), "maximum": int(scaler.data_max_[0]),
            "clip": False,
        },
    }
    identity = {
        "baseline_signature": baseline_signature,
        "dataset": dataset,
        "source_rows": [row["source_row"] for row in rows],
        "config": asdict(config),
        "seed": seed,
        "epochs": epochs_by_variant,
        "micro_batch_size": micro_batch_size,
        "runtime": exp.runtime_info(device),
        "sources": {
            **study.source_hashes(),
            "length_ablation.py": exp.sha256(Path(__file__).with_name("length_ablation.py")),
            "length_ablation_test.py": exp.sha256(__file__),
        },
    }
    signature = exp.fingerprint(identity)
    directory = Path(output_dir) / dataset / f"refit_{signature[:16]}"
    checkpoints = {}
    vocabulary = encoded = None
    for variant, epochs in epochs_by_variant.items():
        path = directory / f"{variant}.pt"
        state = feature_states[variant]
        expected = exp.fingerprint({"refit": signature, "variant": variant, "feature_state": state})
        if path.is_file():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["signature"] != expected:
                raise ValueError(f"Stored refit does not match the current settings: {path}")
            if progress:
                progress(f"Reusing refit for {variant}")
        else:
            if vocabulary is None:
                vocabulary = study.build_vocabulary(rows, config.preprocessing)
                encoded = _encode(rows, vocabulary, config)
            exp.seed_everything(seed)
            model = _LengthCNN(vocabulary, config).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
            )
            loader = _loader(encoded, _features(counts, state), micro_batch_size, seed, shuffle=True)
            history = []
            for epoch in range(1, epochs + 1):
                metrics = study.train_epoch(model, loader, optimizer, device)
                history.append({"epoch": epoch, **metrics})
                if progress:
                    progress(f"Refit {variant}, epoch {epoch}/{epochs}, loss {metrics['train_loss']:.5f}")
            _save_checkpoint(path, {
                "format": "cnn_length_ablation_refit.v1",
                "signature": expected,
                "variant": variant,
                "feature_state": state,
                "config": asdict(config),
                "vocabulary": vocabulary,
                "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "history": history,
                "refit": identity,
            })
            del model, optimizer, loader
        checkpoints[variant] = path
    return directory, checkpoints


def _load_prior_test(baseline_dir, manifest, data_dirs):
    frozen = exp.read_json(baseline_dir / "analysis" / "frozen_evaluation.json")
    metadata = frozen["metadata"]
    if (exp.fingerprint(metadata) != frozen["freeze_signature"]
            or metadata["run_signature"] != manifest["run_signature"]):
        raise ValueError("The baseline's saved test setup does not match its study manifest.")
    identity = {
        "freeze_signature": frozen["freeze_signature"],
        "test_sha256": {
            dataset: exp.sha256(data_dir / "test.json")
            for dataset, data_dir in data_dirs.items()
        },
    }
    directory = baseline_dir / "analysis" / f"test_{exp.fingerprint(identity)[:12]}"
    report = exp.read_json(directory / "report.json")
    if report["identity"] != identity:
        raise ValueError("The saved test comparison uses different test data.")
    return metadata, report


def _aligned_scores(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("Expected one score per test row.")
    scores = []
    for row, prediction in zip(rows, predictions, strict=True):
        if (row["source_row"], row["label"]) != (prediction["source_row"], prediction["label"]):
            raise ValueError("A saved prediction could not be paired with its test row.")
        scores.append(prediction["injection_score"])
    scores = np.array(scores)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected finite test probabilities between 0 and 1.")
    return scores


def _score_refit(path, rows, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != "cnn_length_ablation_refit.v1":
        raise ValueError(f"Unknown length ablation checkpoint format: {path}")
    config = study.config_from_dict(checkpoint["config"])
    vocabulary = checkpoint["vocabulary"]
    model = _LengthCNN(vocabulary, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    encoded = _encode(rows, vocabulary, config)
    numeric = _features(length_counts(rows, "log_tokens"), checkpoint["feature_state"])
    loader = _loader(encoded, numeric, checkpoint["refit"]["micro_batch_size"], 42)
    scores, _ = study.score_loader(model, loader, device)
    return [
        {"source_row": row["source_row"], "label": row["label"], "injection_score": float(score)}
        for row, score in zip(rows, scores, strict=True)
    ]


def evaluate_length_test(
    output_dir, checkpoints, *, baseline_dir, baseline_manifest, data_dirs, dataset, device,
):
    """Compare refits with saved models at threshold 0.5 on matched test cohorts."""
    device = exp.resolve_device(device)
    baseline_dir = Path(baseline_dir)
    frozen, prior = _load_prior_test(baseline_dir, baseline_manifest, data_dirs)
    identity = {
        "baseline_test": prior["identity"],
        "baseline_predictions": prior["predictions"],
        "checkpoint_sha256": {name: exp.sha256(path) for name, path in checkpoints.items()},
        "dataset": dataset,
        "threshold": 0.5,
        "runtime": exp.runtime_info(device),
    }
    report_path = Path(output_dir) / f"test_{exp.fingerprint(identity)[:16]}.json"
    if report_path.is_file():
        report = exp.read_json(report_path)
        if report["identity"] != identity:
            raise ValueError("The saved ablation report belongs to a different test comparison.")
        return report_path, report

    test = study.load_datasets(data_dirs["raw"], data_dirs["normalized"], "test")
    common_ids = {row["source_row"] for row in test["raw"]} & {
        row["source_row"] for row in test["normalized"]
    }
    training_keys = set(frozen["training_collision_keys"])
    eligible = {
        row["source_row"] for row in test["raw"] if row["source_row"] in common_ids
        and study.collision_key(original_text(row)) not in training_keys
    }
    rows_by_model, scores_by_model = {}, {}
    for name, spec in prior["predictions"].items():
        path = baseline_dir.joinpath(*PureWindowsPath(spec["path"]).parts)
        if exp.sha256(path) != spec["sha256"]:
            raise ValueError(f"The saved baseline predictions have changed: {path}")
        with path.open(encoding="utf-8") as stream:
            predictions = [json.loads(line) for line in stream]
        rows = test[frozen["models"][name]["dataset"]]
        rows_by_model[name] = rows
        scores_by_model[name] = _aligned_scores(rows, predictions)

    new_predictions = {}
    for variant, checkpoint in checkpoints.items():
        rows = test[dataset]
        predictions = _score_refit(checkpoint, rows, device)
        new_predictions[variant] = predictions
        rows_by_model[variant] = rows
        scores_by_model[variant] = _aligned_scores(rows, predictions)

    primary = f"{dataset}_selected"
    direct_models = (primary, *checkpoints)
    reports, length_bands = [], []
    prior_defaults = {
        (entry["model"], entry["split"]): entry
        for entry in prior["reports"] if entry["profile"] == "default"
    }
    for model, scores in scores_by_model.items():
        rows = rows_by_model[model]
        labels = np.array([row["label"] for row in rows])
        common = np.array([row["source_row"] in eligible for row in rows])
        if int(common.sum()) != len(eligible):
            raise ValueError(f"{model} is missing rows in the common test cohort.")
        subsets = {"test_common_disjoint_known_benchmark": common}
        if model in direct_models:
            subsets["test_full_known_benchmark"] = np.ones(len(rows), dtype=bool)
        for split, mask in subsets.items():
            metrics = exp.binary_metrics(labels[mask], scores[mask], threshold=0.5)
            previous = prior_defaults.get((model, split))
            if previous and metrics["confusion_matrix"] != previous["confusion_matrix"]:
                raise ValueError(f"{model} does not reproduce the saved test results for {split}.")
            reports.append({"model": model, "split": split, **metrics})
        if model in direct_models:
            common_rows = [row for row, keep in zip(rows, common, strict=True) if keep]
            length_bands.extend(
                {"model": model, **band}
                for band in study.length_slice_report(
                    common_rows, scores[common], frozen["training_length_quartiles"],
                )
            )

    report = {
        "identity": identity,
        "primary_baseline": primary,
        "common_rows": len(eligible),
        "training_length_quartiles": frozen["training_length_quartiles"],
        "reports": reports,
        "length_bands": length_bands,
        "predictions": new_predictions,
    }
    _write_result(report_path, report)
    return report_path, report
