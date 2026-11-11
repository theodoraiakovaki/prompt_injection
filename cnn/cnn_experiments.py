"""Leakage-aware CNN experiments shared by the tuning and analysis notebooks.

Importing this module never starts training. Shared datasets remain at repository root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CNN_DIR = Path(__file__).resolve().parent
ROOT = CNN_DIR.parent
ARTIFACTS_DIR = CNN_DIR / "artifacts"
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PREPROCESSING = {
    "tokenizer": "lowercase_regex_words_and_punctuation_v1",
    "max_vocabulary_size": 20_000,
    "max_sequence_length": 1_024,
    "padding_id": 0,
    "unknown_id": 1,
}
METRICS = (
    "loss", "accuracy", "macro_f1", "injection_precision", "injection_recall",
    "injection_f1", "false_positive_rate", "roc_auc", "average_precision",
)


@dataclass(frozen=True)
class CNNConfig:
    config_id: str = "cnn01"
    name: str = "Current CNN recipe"
    embedding_dim: int = 64
    num_filters: int = 256
    kernel_size: int = 3
    depth: int = 1
    dropout: float = 0.0
    learning_rate: float = 0.005
    weight_decay: float = 0.0
    batch_size: int = 32

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", self.config_id):
            raise ValueError("config_id must be a simple filename-safe identifier")
        if min(self.embedding_dim, self.num_filters, self.kernel_size, self.depth, self.batch_size) < 1:
            raise ValueError("Layer sizes and batch size must be positive")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid dropout, learning rate, or weight decay")


def default_configurations():
    base = CNNConfig()
    lower_lr = replace(base, config_id="cnn02", name="Lower learning rate", learning_rate=0.001)
    dropout = replace(lower_lr, config_id="cnn03", name="Dropout", dropout=0.3)
    regularized = replace(dropout, config_id="cnn04", name="Weight decay", weight_decay=0.0001)
    return [
        base, lower_lr, dropout, regularized,
        replace(regularized, config_id="cnn05", name="128 filters", num_filters=128),
        replace(regularized, config_id="cnn06", name="Kernel size 5", kernel_size=5),
        replace(regularized, config_id="cnn07", name="Two convolutions", depth=2),
        replace(regularized, config_id="cnn08", name="Three convolutions", depth=3),
    ]


@dataclass(frozen=True)
class SearchSettings:
    n_splits: int = 5
    early_stopping_splits: int = 10
    max_epochs: int = 10
    patience: int = 2
    seed: int = 42
    smoke: bool = False
    smoke_rows: int = 160

    def __post_init__(self):
        if min(self.n_splits, self.early_stopping_splits) < 2:
            raise ValueError("At least two folds are required")
        if min(self.max_epochs, self.patience) < 1 or self.smoke_rows < 40:
            raise ValueError("Invalid epoch, patience, or smoke sample count")


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def normalize(text):
    """For grouping only; never substituted for the model input."""
    return " ".join(text.casefold().split())


def group_id(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_predictions(path, rows, scores):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row, score in zip(rows, scores, strict=True):
            output.write(json.dumps({
                "row_index": row["row_index"], "label": row["label"],
                "injection_score": float(score), "prediction": int(score >= 0.5),
            }, allow_nan=False) + "\n")
    temporary.replace(path)


def read_split(data_dir, split):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    source = read_json(Path(data_dir) / f"{split}.json")
    if not isinstance(source, list) or not source:
        raise ValueError(f"{split} must contain a nonempty list")
    rows = []
    for index, row in enumerate(source):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{split} row {index}: missing or empty prompt")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"{split} row {index}: invalid label")
        rows.append({"row_index": index, "prompt": row["prompt"], "label": row["label"]})
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError(f"{split} must contain both classes")
    return rows


def smoke_subset(rows, limit=160, seed=42):
    """Approximately balanced sample, retaining complete normalized groups."""
    grouped = {}
    for row in rows:
        grouped.setdefault(group_id(row["prompt"]), []).append(row)
    if any(len({row["label"] for row in group}) != 1 for group in grouped.values()):
        raise ValueError("Normalized duplicate group has conflicting labels")
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        candidates = [group for group in grouped.values() if group[0]["label"] == label]
        rng.shuffle(candidates)
        count = 0
        for group in candidates:
            selected.extend(group)
            count += len(group)
            if count >= math.ceil(limit / 2):
                break
    return sorted(selected, key=lambda row: row["row_index"])


def make_folds(rows, settings):
    labels = np.array([row["label"] for row in rows])
    groups = np.array([group_id(row["prompt"]) for row in rows])
    # Reject contradictory grouping rather than silently changing labels.
    group_labels = {}
    for group, label in zip(groups, labels, strict=True):
        if group in group_labels and group_labels[group] != label:
            raise ValueError("Normalized duplicate group has conflicting labels")
        group_labels[group] = label
    if min(Counter(group_labels.values()).values()) < settings.n_splits:
        raise ValueError("Too few groups per class for the requested outer folds")
    splitter = StratifiedGroupKFold(settings.n_splits, shuffle=True, random_state=settings.seed)
    folds = []
    for number, (outer_train, heldout) in enumerate(splitter.split(labels, labels, groups), 1):
        inner_groups = {groups[i]: int(labels[i]) for i in outer_train}
        if min(Counter(inner_groups.values()).values()) < settings.early_stopping_splits:
            raise ValueError("Too few groups per class for the early-stopping split")
        inner = StratifiedGroupKFold(
            settings.early_stopping_splits, shuffle=True, random_state=settings.seed + 1000 + number,
        )
        fit, stop = next(inner.split(labels[outer_train], labels[outer_train], groups[outer_train]))
        partitions = {
            "train": outer_train[fit].tolist(), "early_stopping": outer_train[stop].tolist(),
            "heldout": heldout.tolist(),
        }
        role_groups = [{groups[i] for i in indices} for indices in partitions.values()]
        for i, left in enumerate(role_groups):
            for right in role_groups[i + 1:]:
                if left & right:
                    raise AssertionError("Duplicate group crossed a fold boundary")
        if any(set(labels[indices]) != {0, 1} for indices in partitions.values()):
            raise ValueError("A fold partition is missing a class")
        folds.append({"fold": number, "seed": settings.seed + number, **partitions})
    coverage = Counter(i for fold in folds for i in fold["heldout"])
    if coverage != Counter(range(len(rows))):
        raise AssertionError("Outer holdouts must cover every row exactly once")
    return folds


def build_vocabulary(texts):
    counts = Counter(token for text in texts for token in tokenize(text))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(PREPROCESSING["max_vocabulary_size"] - 2):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_texts(texts, vocabulary):
    length = PREPROCESSING["max_sequence_length"]
    result = torch.zeros((len(texts), length), dtype=torch.long)
    for index, text in enumerate(texts):
        ids = [vocabulary.get(token, 1) for token in tokenize(text)[:length]]
        if ids:
            result[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return result


class ExperimentCNN(nn.Module):
    def __init__(self, vocabulary, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(len(vocabulary), config.embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(config.embedding_dim if i == 0 else config.num_filters,
                      config.num_filters, config.kernel_size)
            for i in range(config.depth)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.num_filters, 1)

    def forward(self, token_ids):
        # Empty or very short inputs have no complete convolution window. Their
        # pooled features are zero, leaving a finite classifier-bias prediction.
        minimum_length = 1 + self.config.depth * (self.config.kernel_size - 1)
        if token_ids.shape[1] < minimum_length:
            token_ids = nn.functional.pad(token_ids, (0, minimum_length - token_ids.shape[1]))
        valid = token_ids.ne(0)
        features = self.embedding(token_ids).transpose(1, 2)
        for convolution in self.convolutions:
            valid = valid.unfold(1, self.config.kernel_size, 1).all(dim=-1)
            features = torch.relu(convolution(features))
            features = features.masked_fill(~valid.unsqueeze(1), 0.0)
        pooled = features.amax(dim=2)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA/ROCm device is unavailable")
    return device


def runtime_info(device):
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "hip": torch.version.hip, "cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(rows, vocabulary, config, shuffle=False, seed=42):
    data = TensorDataset(
        encode_texts([row["prompt"] for row in rows], vocabulary),
        torch.tensor([row["label"] for row in rows], dtype=torch.float32),
    )
    return DataLoader(data, batch_size=config.batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_function = nn.BCEWithLogitsLoss()
    total_loss, correct, count = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_function(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        correct += ((logits.detach() >= 0) == targets.bool()).sum().item()
        count += len(targets)
    return {"train_loss": total_loss / count, "train_accuracy": correct / count}


@torch.inference_mode()
def score_loader(model, loader, device):
    model.eval()
    total_loss, scores, count = 0.0, [], 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum").item()
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        count += len(targets)
    return np.array(scores, dtype=float), total_loss / count


def predict_rows(model, rows, vocabulary, config, device="cpu"):
    return score_loader(model, make_loader(rows, vocabulary, config), resolve_device(device))


def _binary_arrays(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels and scores must be nonempty matching vectors")
    if not set(labels).issubset({0, 1}) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected binary labels and finite scores between zero and one")
    return labels, scores


def binary_metrics(labels, scores, threshold=0.5, loss=None):
    labels, scores = _binary_arrays(labels, scores)
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    predictions = (scores >= threshold).astype(int)
    report = classification_report(labels, predictions, labels=[0, 1],
                                   target_names=["benign", "injection"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    clipped = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
    benign_count = tn + fp
    fpr = fp / benign_count if benign_count else None
    # Wilson interval exposes uncertainty at low observed false-positive counts.
    interval = None
    if benign_count:
        z, n = 1.959963984540054, benign_count
        center = (fpr + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(fpr * (1 - fpr) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        interval = [max(0.0, center - half), min(1.0, center + half)]
    both = len(set(labels)) == 2
    return {
        "rows": len(labels), "benign_count": benign_count, "injection_count": tp + fn,
        "loss": float(loss) if loss is not None else float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": report["macro avg"]["f1-score"],
        "injection_precision": report["injection"]["precision"],
        "injection_recall": report["injection"]["recall"],
        "injection_f1": report["injection"]["f1-score"],
        "false_positive_rate": fpr, "fpr_wilson_95": interval,
        "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
        "average_precision": float(average_precision_score(labels, scores)) if both else None,
        "true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp,
        "confusion_matrix": matrix.tolist(), "classification_report": report,
    }


def fit_early_stopping(config, vocabulary, fit_rows, stop_rows, settings, seed, device, progress=print):
    seed_everything(seed)
    model = ExperimentCNN(vocabulary, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    fit_loader = make_loader(fit_rows, vocabulary, config, shuffle=True, seed=seed)
    stop_loader = make_loader(stop_rows, vocabulary, config)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history, started = [], time.perf_counter()
    for epoch in range(1, settings.max_epochs + 1):
        training = train_epoch(model, fit_loader, optimizer, device)
        scores, stop_loss = score_loader(model, stop_loader, device)
        stop_metrics = binary_metrics([row["label"] for row in stop_rows], scores, loss=stop_loss)
        history.append({"epoch": epoch, **training, "early_stopping_loss": stop_loss,
                        "early_stopping_macro_f1": stop_metrics["macro_f1"]})
        if progress:
            progress(f"  epoch {epoch}: train loss {training['train_loss']:.4f}, early-stop loss {stop_loss:.4f}")
        if stop_loss < best_loss:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, time.perf_counter() - started


def save_checkpoint(path, model, vocabulary, config, **metadata):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": "cnn_experiments.v1", "config": asdict(config),
        "preprocessing": PREPROCESSING, "vocabulary": vocabulary,
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        **metadata,
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    device = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") == "cnn_experiments.v1":
        if checkpoint["preprocessing"] != PREPROCESSING:
            raise ValueError("Checkpoint preprocessing differs from the fixed experiment preprocessing")
        config, state = CNNConfig(**checkpoint["config"]), checkpoint["model_state_dict"]
    else:
        # Read a historical one-layer checkpoint without the retired training script.
        original = checkpoint["config"]
        if original["max_sequence_length"] != PREPROCESSING["max_sequence_length"]:
            raise ValueError("Original checkpoint uses a different sequence length")
        config = CNNConfig(config_id="legacy", name="Existing CNN", embedding_dim=original["embedding_dim"],
                           num_filters=original["num_filters"], kernel_size=original["kernel_size"])
        state = {key.replace("convolution.", "convolutions.0."): value
                 for key, value in checkpoint["model_state_dict"].items()}
    model = ExperimentCNN(checkpoint["vocabulary"], config)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint["vocabulary"], config, checkpoint


def _verify_artifact(path, expected_hash):
    if not Path(path).is_file() or sha256(path) != expected_hash:
        raise ValueError(f"Missing or changed experiment artifact: {path}")


def summarize_folds(results, configurations, fold_count):
    summary = []
    for config in configurations:
        entries = [result for result in results if result["config_id"] == config.config_id]
        if len(entries) != fold_count:
            raise ValueError("Cannot rank configurations before every requested fold is complete")
        row = {"config_id": config.config_id, "name": config.name, "folds": len(entries),
               "parameter_count": float(np.mean([entry["parameter_count"] for entry in entries])),
               "training_seconds": sum(entry["training_seconds"] for entry in entries)}
        for metric in METRICS:
            values = [entry["metrics"][metric] for entry in entries]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return sorted(summary, key=lambda row: (-row["macro_f1_mean"], row["macro_f1_std"],
                                            row["parameter_count"], row["config_id"]))


def run_search(run_dir, data_dir=ROOT / "data", settings=None, configurations=None, device="auto", progress=print, refit=True):
    """Train-only search; official validation and test files are never opened."""
    settings = settings or SearchSettings()
    configurations = list(configurations or default_configurations())
    if len({config.config_id for config in configurations}) != len(configurations):
        raise ValueError("Configuration identifiers must be unique")
    rows = read_split(data_dir, "train")
    if settings.smoke:
        rows = smoke_subset(rows, settings.smoke_rows, settings.seed)
        configurations = configurations[:1]
        settings = replace(settings, max_epochs=1)
    folds = make_folds(rows, settings)
    executed_folds = folds[:1] if settings.smoke else folds
    device, run_dir = resolve_device(device), Path(run_dir).resolve()
    recipe = {
        "format": "cnn_search.v1", "mode": "smoke" if settings.smoke else "full",
        "settings": asdict(settings), "configurations": [asdict(config) for config in configurations],
        "preprocessing": PREPROCESSING, "training_sha256": sha256(Path(data_dir) / "train.json"),
        "source_sha256": sha256(__file__), "runtime": runtime_info(device),
        "row_indices": [row["row_index"] for row in rows], "folds": folds,
        "executed_folds": [fold["fold"] for fold in executed_folds],
    }
    signature = fingerprint(recipe)
    manifest = {**recipe, "run_signature": signature}
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Run directory belongs to a different recipe, data, code, or runtime; choose a new directory")
    write_json(manifest_path, manifest)
    fold_results = []
    for config in configurations:
        for fold in executed_folds:
            directory = run_dir / config.config_id / f"fold_{fold['fold']}"
            directory.mkdir(parents=True, exist_ok=True)
            completion = directory / "result.json"
            if completion.exists():
                result = read_json(completion)
                if result["run_signature"] != signature:
                    raise ValueError("Cached fold does not match this run")
                _verify_artifact(directory / "model.pt", result["checkpoint_sha256"])
                _verify_artifact(directory / "predictions.jsonl", result["predictions_sha256"])
                fold_results.append(result)
                if progress:
                    progress(f"Resume {config.config_id}, fold {fold['fold']}: already complete")
                continue
            if progress:
                progress(f"Training {config.config_id} ({config.name}), fold {fold['fold']}/{settings.n_splits} on {device}")
            fit_rows = [rows[i] for i in fold["train"]]
            stop_rows = [rows[i] for i in fold["early_stopping"]]
            heldout_rows = [rows[i] for i in fold["heldout"]]
            vocabulary = build_vocabulary([row["prompt"] for row in fit_rows])
            model, history, best_epoch, seconds = fit_early_stopping(
                config, vocabulary, fit_rows, stop_rows, settings, fold["seed"], device, progress,
            )
            scores, loss = predict_rows(model, heldout_rows, vocabulary, config, device)
            save_checkpoint(directory / "model.pt", model, vocabulary, config, seed=fold["seed"],
                            best_epoch=best_epoch, history=history, run_signature=signature, fold=fold["fold"])
            write_predictions(directory / "predictions.jsonl", heldout_rows, scores)
            result = {
                "run_signature": signature, "config_id": config.config_id, "fold": fold["fold"],
                "seed": fold["seed"], "best_epoch": best_epoch, "history": history,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "training_seconds": seconds, "runtime": runtime_info(device),
                "metrics": binary_metrics([row["label"] for row in heldout_rows], scores, loss=loss),
                "checkpoint_sha256": sha256(directory / "model.pt"),
                "predictions_sha256": sha256(directory / "predictions.jsonl"),
            }
            write_json(completion, result)  # Completion marker is always written last.
            fold_results.append(result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    leaderboard = summarize_folds(fold_results, configurations, len(executed_folds))
    write_json(run_dir / "leaderboard.json", leaderboard)
    with (run_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(leaderboard[0]))
        writer.writeheader()
        writer.writerows(leaderboard)
    if refit:
        return refit_selected(run_dir, data_dir, device, progress)
    return load_search(run_dir, require_selected=False)


def refit_selected(run_dir, data_dir=ROOT / "data", device="auto", progress=print):
    """Separate, reviewable step after completing and inspecting the CV search."""
    search = load_search(run_dir, require_selected=False)
    if search["selected"] is not None:
        # Loading an already verified checkpoint never needs its original runtime
        # or training source. Actual retraining retains the strict checks below.
        if progress:
            progress("Loaded the verified selected checkpoint; no retraining needed")
        return search
    manifest = search["manifest"]
    device, run_dir = resolve_device(device), Path(run_dir).resolve()
    if runtime_info(device) != manifest["runtime"] or sha256(__file__) != manifest["source_sha256"]:
        raise ValueError("Refit must use the search's recorded code and runtime")
    if sha256(Path(data_dir) / "train.json") != manifest["training_sha256"]:
        raise ValueError("Training data changed after the CNN search")
    all_rows = read_split(data_dir, "train")
    rows = [all_rows[i] for i in manifest["row_indices"]]
    settings = SearchSettings(**manifest["settings"])
    winner = next(CNNConfig(**config) for config in manifest["configurations"]
                  if config["config_id"] == search["leaderboard"][0]["config_id"])
    signature = manifest["run_signature"]
    best_epochs = [result["best_epoch"] for result in search["fold_results"] if result["config_id"] == winner.config_id]
    final_epochs = int(np.median(best_epochs))
    selected_path = run_dir / "selected.json"
    if selected_path.exists():
        selected = read_json(selected_path)
        if selected["run_signature"] != signature or selected["config_id"] != winner.config_id:
            raise ValueError("Selected checkpoint does not match the completed search")
        _verify_artifact(run_dir / "selected.pt", selected["checkpoint_sha256"])
    else:
        if progress:
            progress(f"Retraining {winner.config_id} on all {len(rows):,} training rows for {final_epochs} epoch(s)")
        seed_everything(settings.seed)
        vocabulary = build_vocabulary([row["prompt"] for row in rows])
        model = ExperimentCNN(vocabulary, winner).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=winner.learning_rate, weight_decay=winner.weight_decay)
        loader = make_loader(rows, vocabulary, winner, shuffle=True, seed=settings.seed)
        history, started = [], time.perf_counter()
        for epoch in range(1, final_epochs + 1):
            history.append({"epoch": epoch, **train_epoch(model, loader, optimizer, device)})
        save_checkpoint(run_dir / "selected.pt", model, vocabulary, winner, seed=settings.seed,
                        epochs=final_epochs, history=history, run_signature=signature, mode=manifest["mode"])
        selected = {
            "run_signature": signature, "mode": manifest["mode"], "config_id": winner.config_id,
            "config": asdict(winner), "epochs": final_epochs, "fold_best_epochs": best_epochs,
            "seed": settings.seed, "history": history, "training_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha256(run_dir / "selected.pt"), "runtime": runtime_info(device),
        }
        write_json(selected_path, selected)
    return load_search(run_dir)


def load_search(run_dir, require_selected=True):
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "manifest.json")
    if fingerprint({key: value for key, value in manifest.items() if key != "run_signature"}) != manifest["run_signature"]:
        raise ValueError("Search manifest was changed after its run signature was created")
    selected = read_json(run_dir / "selected.json") if (run_dir / "selected.json").exists() else None
    if require_selected and selected is None:
        raise FileNotFoundError("Run the final-retraining step in cnn_tuning.ipynb first")
    if selected is not None:
        if selected["run_signature"] != manifest["run_signature"]:
            raise ValueError("Selected model and search manifest disagree")
        _verify_artifact(run_dir / "selected.pt", selected["checkpoint_sha256"])
    results = [read_json(run_dir / config["config_id"] / f"fold_{fold}" / "result.json")
               for config in manifest["configurations"] for fold in manifest["executed_folds"]]
    for result in results:
        if result["run_signature"] != manifest["run_signature"]:
            raise ValueError("Fold result belongs to a different search")
    expected = summarize_folds(results, [CNNConfig(**config) for config in manifest["configurations"]],
                               len(manifest["executed_folds"]))
    if read_json(run_dir / "leaderboard.json") != expected:
        raise ValueError("Leaderboard differs from the completed fold results")
    return {"run_dir": str(run_dir), "manifest": manifest, "selected": selected,
            "leaderboard": read_json(run_dir / "leaderboard.json"), "fold_results": results}


def threshold_profiles(labels, scores):
    """Choose operating points on development data, never on test predictions."""
    labels, scores = _binary_arrays(labels, scores)
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection requires both classes")
    fpr, recall, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    # sklearn uses infinity for predicting all benign; retain a finite JSON-safe
    # equivalent, including when a float32 sigmoid has rounded a score to 1.0.
    thresholds[0] = np.nextafter(1.0, 2.0)
    positive, negative = int(labels.sum()), int((labels == 0).sum())
    tp, fp = np.rint(recall * positive), np.rint(fpr * negative)
    fn, tn = positive - tp, negative - fp
    macro = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
    indices = range(len(thresholds))
    balanced = max(indices, key=lambda i: (macro[i], -fpr[i], thresholds[i]))
    low_fpr = max((i for i in indices if fpr[i] <= 0.01),
                  key=lambda i: (recall[i], -fpr[i], thresholds[i]))
    high_recall = min((i for i in indices if recall[i] >= 0.99),
                      key=lambda i: (fpr[i], -recall[i], -thresholds[i]))
    chosen = {"default": 0.5, "balanced": float(thresholds[balanced]),
              "low_false_alarms": float(thresholds[low_fpr]), "high_detection": float(thresholds[high_recall])}
    return {name: {"threshold": threshold, "development_metrics": binary_metrics(labels, scores, threshold)}
            for name, threshold in chosen.items()}


def diagnostics(rows, scores, vocabulary, threshold=0.5):
    """Return reviewable records, retaining prompt text only in notebook memory."""
    result = []
    limit = PREPROCESSING["max_sequence_length"]
    for row, score in zip(rows, scores, strict=True):
        tokens = tokenize(row["prompt"])
        used = tokens[:limit]
        prediction = int(score >= threshold)
        result.append({
            "row_index": row["row_index"], "label": row["label"], "prediction": prediction,
            "injection_score": float(score), "predicted_class_score": float(score if prediction else 1 - score),
            "error_type": "false_positive" if prediction == 1 and row["label"] == 0 else
                          "false_negative" if prediction == 0 and row["label"] == 1 else "correct",
            "characters": len(row["prompt"]), "token_count": len(tokens), "truncated": len(tokens) > limit,
            "unknown_token_rate": sum(token not in vocabulary for token in used) / len(used) if used else 0.0,
            "prompt_preview": row["prompt"][:800],
        })
    return result


def _profile_rows(model_name, split, rows, scores, profiles, mask=None):
    if mask is None:
        mask = np.ones(len(rows), dtype=bool)
    labels = np.array([row["label"] for row in rows])[mask]
    return [{"model": model_name, "split": split, "profile": profile,
             "threshold": value["threshold"],
             **binary_metrics(labels, np.asarray(scores)[mask], value["threshold"])}
            for profile, value in profiles.items()]


def prepare_validation_analysis(run_dir, data_dir=ROOT / "data", device="auto", legacy_checkpoint=None):
    """Evaluate validation and persist baselines; test data is never opened here."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    search = load_search(run_dir)
    manifest = search["manifest"]
    data_dir, run_dir, device = Path(data_dir), Path(run_dir).resolve(), resolve_device(device)
    if sha256(data_dir / "train.json") != manifest["training_sha256"]:
        raise ValueError("Training data changed after the CNN search")
    all_training = read_split(data_dir, "train")
    training = [all_training[i] for i in manifest["row_indices"]]
    validation = read_split(data_dir, "validation")
    if manifest["mode"] == "smoke":
        validation = smoke_subset(validation, 100, manifest["settings"]["seed"])
    # Check overlap against ALL official training rows, even in smoke mode.
    training_groups = {group_id(row["prompt"]) for row in all_training}
    clean_mask = np.array([group_id(row["prompt"]) not in training_groups for row in validation])
    if set(row["label"] for row, keep in zip(validation, clean_mask, strict=True) if keep) != {0, 1}:
        raise ValueError("Overlap-free validation must contain both classes")
    analysis_dir = run_dir / "analysis"
    historical_report = analysis_dir / "validation_report.json"
    if historical_report.exists() and read_json(historical_report)["metadata"]["source_sha256"] != sha256(__file__):
        # Preserve published results and frozen thresholds after a code/layout change.
        # A fresh evaluation reuses the trained checkpoint in a separate directory.
        analysis_dir = run_dir / ("analysis_" + sha256(__file__)[:12])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_specs = {
        "selected_cnn": {"kind": "cnn", "path": str(run_dir / "selected.pt"),
                         "sha256": sha256(run_dir / "selected.pt")},
    }
    legacy_checkpoint = Path(legacy_checkpoint) if legacy_checkpoint is not None else ARTIFACTS_DIR / "legacy" / "cnn_v1_seed42.pt"
    if legacy_checkpoint.is_file():
        model_specs["existing_cnn"] = {"kind": "cnn", "path": str(legacy_checkpoint.resolve()),
                                        "sha256": sha256(legacy_checkpoint)}

    tfidf_path, tfidf_meta = analysis_dir / "tfidf.joblib", analysis_dir / "tfidf.json"
    tfidf_recipe = {
        "training_sha256": manifest["training_sha256"], "training_indices": manifest["row_indices"],
        "max_features": 20_000, "max_iter": 1000, "random_state": 42,
        "scikit_learn": sklearn.__version__,
    }
    if tfidf_meta.exists():
        stored = read_json(tfidf_meta)
        if stored["recipe"] != tfidf_recipe:
            raise ValueError("TF-IDF artifact belongs to a different training recipe")
        _verify_artifact(tfidf_path, stored["sha256"])
        tfidf = joblib.load(tfidf_path)
    else:
        tfidf = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20_000)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        tfidf.fit([row["prompt"] for row in training], [row["label"] for row in training])
        joblib.dump(tfidf, tfidf_path)
        write_json(tfidf_meta, {"recipe": tfidf_recipe, "sha256": sha256(tfidf_path)})
    model_specs["tfidf_logistic"] = {"kind": "tfidf", "path": str(tfidf_path), "sha256": sha256(tfidf_path)}

    all_scores, profiles, reports, vocabulary = {}, {}, [], None
    for model_name, spec in model_specs.items():
        if spec["kind"] == "cnn":
            model, model_vocabulary, config, _ = load_checkpoint(spec["path"], device)
            scores, _ = predict_rows(model, validation, model_vocabulary, config, device)
            if model_name == "selected_cnn":
                vocabulary = model_vocabulary
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            scores = tfidf.predict_proba([row["prompt"] for row in validation])[:, 1]
        all_scores[model_name] = scores
        clean_labels = [row["label"] for row, keep in zip(validation, clean_mask, strict=True) if keep]
        profiles[model_name] = threshold_profiles(clean_labels, scores[clean_mask])
        reports.extend(_profile_rows(model_name, "validation_official", validation, scores, profiles[model_name]))
        reports.extend(_profile_rows(model_name, "validation_without_train_overlap", validation, scores,
                                     profiles[model_name], clean_mask))
        write_predictions(analysis_dir / f"{model_name}_validation_predictions.jsonl", validation, scores)

    metadata = {
        "mode": manifest["mode"], "run_signature": manifest["run_signature"],
        "source_sha256": sha256(__file__), "training_sha256": manifest["training_sha256"],
        "validation_sha256": sha256(data_dir / "validation.json"),
        "validation_indices": [row["row_index"] for row in validation],
        "calibration_indices": [row["row_index"] for row, keep in zip(validation, clean_mask, strict=True) if keep],
        "removed_overlap_rows": int((~clean_mask).sum()),
        "models": model_specs, "profiles": profiles, "runtime": runtime_info(device),
        "interpretation": "Threshold-selected validation metrics are development results. Test is a previously inspected benchmark.",
    }
    write_json(analysis_dir / "validation_report.json", {"metadata": metadata, "reports": reports})
    return {"search": search, "rows": validation, "clean_mask": clean_mask, "scores": all_scores,
            "profiles": profiles, "reports": reports, "metadata": metadata,
            "vocabulary": vocabulary, "analysis_dir": str(analysis_dir)}


def freeze_analysis(analysis):
    """Persist immutable model references and thresholds before any test scoring."""
    metadata = analysis["metadata"]
    for spec in metadata["models"].values():
        _verify_artifact(spec["path"], spec["sha256"])
    frozen = {**metadata, "freeze_signature": fingerprint(metadata)}
    path = Path(analysis["analysis_dir"]) / "frozen_evaluation.json"
    if path.exists() and read_json(path) != frozen:
        raise ValueError("This analysis already has a different frozen evaluation; use a new experiment directory")
    write_json(path, frozen)
    return path


def evaluate_frozen_test(frozen_path, data_dir=ROOT / "data", device="auto"):
    """Apply stored thresholds without selecting any settings from test data."""
    import joblib

    frozen_path, data_dir = Path(frozen_path), Path(data_dir)
    frozen = read_json(frozen_path)
    if frozen["mode"] != "full":
        raise ValueError("Smoke artifacts cannot be used to score the official test split")
    unsigned = {key: value for key, value in frozen.items() if key != "freeze_signature"}
    if fingerprint(unsigned) != frozen["freeze_signature"]:
        raise ValueError("Frozen evaluation was edited after freezing")
    if frozen["source_sha256"] != sha256(__file__):
        raise ValueError("Experiment code changed after thresholds were frozen")
    for split, key in (("train", "training_sha256"), ("validation", "validation_sha256")):
        if sha256(data_dir / f"{split}.json") != frozen[key]:
            raise ValueError(f"{split} data changed after freezing")
    for spec in frozen["models"].values():
        _verify_artifact(spec["path"], spec["sha256"])
    device = resolve_device(device)
    identity = {"freeze_signature": frozen["freeze_signature"], "test_sha256": sha256(data_dir / "test.json"),
                "runtime": runtime_info(device)}
    directory = frozen_path.parent / ("test_" + fingerprint(identity)[:16])
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "report.json"
    if completed.exists():
        cached = read_json(completed)
        if cached["identity"] != identity:
            raise ValueError("Cached test evaluation identity mismatch")
        for name, digest in cached["predictions_sha256"].items():
            _verify_artifact(directory / f"{name}_predictions.jsonl", digest)
        return cached
    rows, reports, prediction_hashes = read_split(data_dir, "test"), [], {}
    for model_name, spec in frozen["models"].items():
        if spec["kind"] == "cnn":
            model, vocabulary, config, _ = load_checkpoint(spec["path"], device)
            scores, _ = predict_rows(model, rows, vocabulary, config, device)
            del model
        else:
            model = joblib.load(spec["path"])
            scores = model.predict_proba([row["prompt"] for row in rows])[:, 1]
        reports.extend(_profile_rows(model_name, "test_known_benchmark", rows, scores, frozen["profiles"][model_name]))
        prediction_path = directory / f"{model_name}_predictions.jsonl"
        write_predictions(prediction_path, rows, scores)
        prediction_hashes[model_name] = sha256(prediction_path)
    report = {"identity": identity, "reports": reports, "predictions_sha256": prediction_hashes,
              "directory": str(directory.resolve()), "interpretation": "Known benchmark; no test-based tuning."}
    write_json(completed, report)
    return report
