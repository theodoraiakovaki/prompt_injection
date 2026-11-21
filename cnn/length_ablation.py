"""Train the paired prompt-length variants using the saved CNN study folds."""

from dataclasses import asdict
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import cnn_experiments as exp
from . import cnn_study as study
from .length_features import LengthState


class _LengthCNN(study.StudyCNN):
    def __init__(self, vocabulary, config):
        super().__init__(vocabulary, config, LengthState())
        text_head = self.classifier
        self.numeric_dim = 1
        self.classifier = nn.Linear(text_head.in_features + 1, text_head.out_features)
        with torch.no_grad():
            self.classifier.weight[:, :-1].copy_(text_head.weight)
            self.classifier.weight[:, -1].zero_()
            self.classifier.bias.copy_(text_head.bias)


def _loader(encoded, numeric, batch_size, seed, *, shuffle=False):
    token_ids, labels = encoded
    numeric = torch.from_numpy(numeric)
    if numeric.shape != (len(labels), 1) or numeric.dtype != torch.float32:
        raise ValueError("Expected one float32 length feature per row.")
    dataset = TensorDataset(token_ids, numeric, labels)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def _train(config, settings, vocabulary, encoded, features, seed, micro_batch, device, progress):
    exp.seed_everything(seed)
    model = _LengthCNN(vocabulary, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    fit = _loader(encoded["train"], features["train"], micro_batch, seed, shuffle=True)
    stop = _loader(encoded["early_stopping"], features["early_stopping"], micro_batch, seed)
    heldout = _loader(encoded["heldout"], features["heldout"], micro_batch, seed)

    best_loss = math.inf
    history = []
    for epoch in range(1, settings.max_epochs + 1):
        training_metrics = study.train_epoch(model, fit, optimizer, device)
        _, stop_loss = study.score_loader(model, stop, device)
        if not math.isfinite(stop_loss):
            raise RuntimeError("Non-finite early-stopping loss.")
        history.append({"epoch": epoch, **training_metrics, "early_stopping_loss": stop_loss})
        if progress:
            progress(f"  epoch {epoch}: early-stopping loss {stop_loss:.5f}")
        if stop_loss < best_loss:
            best_loss, best_epoch = stop_loss, epoch
            best_weights = {name: weights.detach().cpu().clone()
                            for name, weights in model.state_dict().items()}
        if epoch - best_epoch >= settings.patience:
            break

    model.load_state_dict(best_weights)
    scores, loss = study.score_loader(model, heldout, device)
    return scores, loss, best_epoch, history


def _write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temporary_path = Path(output.name)
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(result, output, allow_nan=False, default=exp._json_default)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_ablation(
    output_dir, config, settings, fold_partitions, fold_features, fold_scalers,
    *, dataset, baseline_signature, seed, micro_batch_sizes, device, progress=print,
):
    """Train fresh models and persist each fold's held-out predictions and metrics."""
    device = exp.resolve_device(device)
    run_identity = {
        "baseline_signature": baseline_signature,
        "dataset": dataset,
        "seed": seed,
        "config": asdict(config),
        "settings": asdict(settings),
        "runtime": exp.runtime_info(device),
        "sources": {**study.source_hashes(), "length_ablation.py": exp.sha256(__file__)},
        "micro_batch_sizes": micro_batch_sizes,
    }
    run_dir = Path(output_dir) / dataset / f"seed_{seed}_{exp.fingerprint(run_identity)[:16]}"
    results = []
    for fold_id, parts in fold_partitions.items():
        micro_batch = micro_batch_sizes[fold_id]
        if micro_batch < 1 or config.batch_size % micro_batch:
            raise ValueError(f"Invalid microbatch size for fold {fold_id}: {micro_batch}.")
        encoded = None
        for variant, features in fold_features[fold_id].items():
            path = run_dir / variant / f"fold_{fold_id}.json"
            scaler = fold_scalers[fold_id]
            feature_state = (
                {"minimum": float(scaler.data_min_[0]), "maximum": float(scaler.data_max_[0]),
                 "clip": bool(scaler.clip)}
                if variant == "minmax_tokens" else {}
            )
            identity = exp.fingerprint({
                "run": run_identity,
                "variant": variant,
                "fold": fold_id,
                "feature_state": feature_state,
                "rows": {
                    role: [(row["source_row"], row["label"]) for row in rows]
                    for role, rows in parts.items()
                },
                "feature_values": {role: values.tolist() for role, values in features.items()},
            })
            if path.exists():
                result = exp.read_json(path)
                if result["identity"] != identity:
                    raise ValueError(f"Saved fold differs from the current inputs: {path}")
                results.append(result)
                if progress:
                    progress(f"Reusing {variant}, fold {fold_id}, seed {seed}")
                continue

            if encoded is None:
                vocabulary = study.build_vocabulary(parts["train"], config.preprocessing)
                encoded = {
                    role: (
                        study.encode_rows(rows, vocabulary, config.preprocessing),
                        torch.tensor([row["label"] for row in rows], dtype=torch.float32),
                    )
                    for role, rows in parts.items()
                }
            if progress:
                progress(f"Training {variant}, fold {fold_id}, seed {seed} on {device}")
            started = time.perf_counter()
            scores, loss, best_epoch, history = _train(
                config, settings, vocabulary, encoded, features, seed, micro_batch, device, progress,
            )
            predictions = [
                {"source_row": row["source_row"], "label": row["label"],
                 "injection_score": float(score)}
                for row, score in zip(parts["heldout"], scores, strict=True)
            ]
            metrics = exp.binary_metrics(
                [row["label"] for row in parts["heldout"]], scores, loss=loss,
            )
            result = {
                "identity": identity, "variant": variant, "fold": fold_id, "seed": seed,
                "feature_state": feature_state, "best_epoch": best_epoch, "history": history,
                "training_seconds": time.perf_counter() - started,
                "metrics": metrics, "predictions": predictions,
            }
            _write_result(path, result)
            results.append(result)
            if progress:
                progress(f"  saved {variant} fold {fold_id}: macro F1 {metrics['macro_f1']:.2%}")

    return run_dir, results
