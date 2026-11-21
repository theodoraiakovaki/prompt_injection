"""Paired CNN study: dataset cleaning, text processing, length and capacity.

Importing this module performs no training and opens no datasets. The historical
v1 search remains in cnn_experiments; this module shares its metrics and IO.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import random
import time
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from . import cnn_experiments as base
    from .length_features import LengthState, PreprocessingConfig, length_counts, original_text, tokenize
except ImportError:  # Support opening notebooks with cnn/ as the working directory.
    import cnn_experiments as base
    from length_features import LengthState, PreprocessingConfig, length_counts, original_text, tokenize


@dataclass(frozen=True)
class StudyConfig(base.CNNConfig):
    config_id: str = "reference"
    name: str = "cnn06 reference"
    kernel_size: int = 5
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    kernel_sizes: tuple[int, ...] = ()
    length_feature: str = "none"

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.preprocessing, PreprocessingConfig):
            raise ValueError("Expected a PreprocessingConfig")
        if self.batch_size != 32:
            raise ValueError("The study fixes the effective training batch size at 32")
        if self.kernel_sizes and (self.depth != 1 or min(self.kernel_sizes) < 1):
            raise ValueError("Parallel kernels require depth one and positive widths")
        if self.length_feature not in {"none", "log_tokens", "log_characters"}:
            raise ValueError("Unknown length feature")


def config_from_dict(value):
    value = dict(value)
    value["preprocessing"] = PreprocessingConfig(**value["preprocessing"])
    value["kernel_sizes"] = tuple(value.get("kernel_sizes", ()))
    return StudyConfig(**value)


@dataclass(frozen=True)
class StudySettings:
    n_splits: int = 5
    early_stopping_splits: int = 10
    split_seed: int = 42
    training_seeds: tuple[int, ...] = (42, 123, 2026)
    max_epochs: int = 10
    patience: int = 2
    smoke: bool = False
    smoke_rows: int = 160

    def __post_init__(self):
        if min(self.n_splits, self.early_stopping_splits) < 2:
            raise ValueError("At least two split groups are required")
        if self.training_seeds != (42, 123, 2026):
            raise ValueError("This study uses confirmation seeds 42, 123, 2026")
        if min(self.max_epochs, self.patience) < 1 or self.smoke_rows < 40:
            raise ValueError("Invalid training settings")


def screening_configs():
    anchor = StudyConfig()
    return [anchor,
            replace(anchor, config_id="length_tokens", name="Token length", length_feature="log_tokens"),
            replace(anchor, config_id="length_characters", name="Character length", length_feature="log_characters"),
            replace(anchor, config_id="vocabulary_40k", name="40k vocabulary",
                    preprocessing=replace(anchor.preprocessing, max_vocabulary_size=40_000)),
            replace(anchor, config_id="embedding_128", name="128 dimensional embeddings", embedding_dim=128),
            replace(anchor, config_id="filters_512", name="512 filters", num_filters=512),
            replace(anchor, config_id="depth_2", name="Two convolutions", depth=2),
            replace(anchor, config_id="depth_3", name="Three convolutions", depth=3),
            replace(anchor, config_id="kernel_7", name="Kernel width 7", kernel_size=7),
            replace(anchor, config_id="parallel_357", name="Parallel kernels 3/5/7",
                    kernel_sizes=(3, 5, 7), num_filters=128),
            replace(anchor, config_id="window_2048", name="2048 token input",
                    preprocessing=replace(anchor.preprocessing, max_sequence_length=2048))]


def controlled_configs():
    return [replace(StudyConfig(), config_id=f"case_{int(lowercase)}_{mode}",
                    name=f"lowercase={lowercase}, punctuation={mode}",
                    preprocessing=PreprocessingConfig(lowercase=lowercase, punctuation_mode=mode))
            for lowercase in (False, True)
            for mode in ("keep", "remove_bang_question", "words_only")]


def collision_key(text):
    """Strongest tested loss of information; covers every tokenizer collision."""
    return base.fingerprint(tokenize(text.casefold(), PreprocessingConfig(punctuation_mode="words_only")))


def map_normalized_rows(raw_rows, normalized_rows):
    """Match normalized records to the earliest available original occurrence."""
    candidates = defaultdict(deque)
    for row in raw_rows:
        candidates[(row["prompt"].lower(), row["label"])].append(row)
    mapped = []
    for row in normalized_rows:
        if row["prompt"] != row["prompt"].lower():
            raise ValueError("Normalized export contains non-lowercase text")
        matches = candidates[(row["prompt"], row["label"])]
        if not matches:
            raise ValueError("Normalized record cannot be matched to raw data")
        original = matches.popleft()
        mapped.append({**row, "source_row": original["row_index"],
                       "original_prompt": original["prompt"]})
    return mapped


def load_datasets(raw_dir, normalized_dir, split):
    raw = [{**row, "source_row": row["row_index"], "original_prompt": row["prompt"]}
           for row in base.read_split(raw_dir, split)]
    normalized = map_normalized_rows(raw, base.read_split(normalized_dir, split))
    return {"raw": raw, "normalized": normalized,
            "controlled": [{**row, "prompt": row["original_prompt"]} for row in normalized]}


def shared_folds(raw_rows, settings):
    labels = np.array([row["label"] for row in raw_rows])
    groups = np.array([collision_key(original_text(row)) for row in raw_rows])
    splitter = StratifiedGroupKFold(settings.n_splits, shuffle=True, random_state=settings.split_seed)
    folds = []
    for number, (outer, heldout) in enumerate(splitter.split(labels, labels, groups), 1):
        inner = StratifiedGroupKFold(settings.early_stopping_splits, shuffle=True,
                                    random_state=settings.split_seed + 1000 + number)
        fit, stop = next(inner.split(labels[outer], labels[outer], groups[outer]))
        indices = {"train": outer[fit], "early_stopping": outer[stop], "heldout": heldout}
        partitions = {role: [raw_rows[int(i)]["source_row"] for i in values]
                      for role, values in indices.items()}
        if any(set(labels[values]) != {0, 1} for values in indices.values()):
            raise ValueError("A shared partition is missing a class; increase the sample")
        folds.append({"fold": number, **partitions})
    return folds


def partition_rows(rows, fold):
    by_id = {row["source_row"]: row for row in rows}
    partitions = {role: [by_id[i] for i in fold[role] if i in by_id]
                  for role in ("train", "early_stopping", "heldout")}
    if any({r["label"] for r in values} != {0, 1} for values in partitions.values()):
        raise ValueError("Dataset restriction removed a class from a fold partition")
    return partitions


def build_vocabulary(rows, preprocessing):
    counts = Counter(token for row in rows for token in tokenize(row["prompt"], preprocessing))
    vocabulary = {base.PAD_TOKEN: 0, base.UNK_TOKEN: 1}
    for token, _ in counts.most_common(preprocessing.max_vocabulary_size - 2):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_rows(rows, vocabulary, preprocessing):
    encoded = torch.zeros((len(rows), preprocessing.max_sequence_length), dtype=torch.long)
    for index, row in enumerate(rows):
        ids = [vocabulary.get(t, 1) for t in tokenize(row["prompt"], preprocessing)
               [:preprocessing.max_sequence_length]]
        if ids:
            encoded[index, :len(ids)] = torch.tensor(ids)
    return encoded


class StudyCNN(nn.Module):
    def __init__(self, vocabulary, config, length_state):
        super().__init__()
        if length_state.mode != config.length_feature:
            raise ValueError("Checkpoint length state and configuration disagree")
        self.config, self.length_state = config, length_state
        self.embedding = nn.Embedding(len(vocabulary), config.embedding_dim, padding_idx=0)
        widths = config.kernel_sizes or (config.kernel_size,)
        self.branches = nn.ModuleList([nn.ModuleList([
            nn.Conv1d(config.embedding_dim if depth == 0 else config.num_filters,
                      config.num_filters, width) for depth in range(config.depth)
        ]) for width in widths])
        self.dropout = nn.Dropout(config.dropout)
        self.numeric_dim = int(config.length_feature != "none")
        self.classifier = nn.Linear(config.num_filters * len(widths) + self.numeric_dim, 1)

    def forward(self, token_ids, length_features=None):
        pooled = []
        for branch in self.branches:
            minimum = 1 + self.config.depth * (branch[0].kernel_size[0] - 1)
            ids = nn.functional.pad(token_ids, (0, max(0, minimum - token_ids.shape[1])))
            valid = ids.ne(0)
            features = self.embedding(ids).transpose(1, 2)
            for convolution in branch:
                valid = valid.unfold(1, convolution.kernel_size[0], 1).all(dim=-1)
                features = torch.relu(convolution(features)).masked_fill(~valid.unsqueeze(1), 0.0)
            pooled.append(features.amax(dim=2))
        features = self.dropout(torch.cat(pooled, dim=1))
        if self.numeric_dim:
            if length_features is None or length_features.shape != (len(token_ids), 1):
                raise ValueError("Expected one standardized length feature per document")
            features = torch.cat((features, length_features), dim=1)
        return self.classifier(features).squeeze(-1)


def make_loader(rows, vocabulary, model, batch_size=32, shuffle=False, seed=42):
    dataset = TensorDataset(encode_rows(rows, vocabulary, model.config.preprocessing),
                            model.length_state.transform(rows),
                            torch.tensor([row["label"] for row in rows], dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def train_epoch(model, loader, optimizer, device):
    """Sum gradients by example, including the final partial effective batch."""
    model.train()
    loss_sum, correct, count, accumulated = 0.0, 0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for index, (tokens, numeric, labels) in enumerate(loader):
        tokens, numeric, labels = tokens.to(device), numeric.to(device), labels.to(device)
        logits = model(tokens, numeric)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        accumulated += len(labels)
        if accumulated == model.config.batch_size or index + 1 == len(loader):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
        loss_sum += loss.item()
        count += len(labels)
        correct += ((logits.detach() >= 0) == labels.bool()).sum().item()
    return {"train_loss": loss_sum / count, "train_accuracy": correct / count}


@torch.inference_mode()
def score_loader(model, loader, device):
    model.eval()
    scores, loss_sum, count = [], 0.0, 0
    for tokens, numeric, labels in loader:
        tokens, numeric, labels = tokens.to(device), numeric.to(device), labels.to(device)
        logits = model(tokens, numeric)
        loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum").item()
        scores.extend(logits.sigmoid().cpu().tolist())
        count += len(labels)
    return np.array(scores), loss_sum / count


def predict_rows(model, rows, vocabulary, config=None, device="cpu", batch_size=32):
    return score_loader(model, make_loader(rows, vocabulary, model, batch_size), base.resolve_device(device))


def save_checkpoint(path, model, vocabulary, **metadata):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": "cnn_study.v2", "config": asdict(model.config), "vocabulary": vocabulary,
                "length_state": asdict(model.length_state),
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "metadata": metadata}, temporary)
    temporary.replace(path)


def load_checkpoint_payload(checkpoint, device="cpu"):
    if "length_state" not in checkpoint:
        raise ValueError("Study checkpoint is missing fitted length state")
    config = config_from_dict(checkpoint["config"])
    model = StudyCNN(checkpoint["vocabulary"], config, LengthState(**checkpoint["length_state"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(base.resolve_device(device)).eval()
    return model, checkpoint["vocabulary"], config, checkpoint


def source_hashes():
    return {name: base.sha256(Path(__file__).parent / name)
            for name in ("cnn_experiments.py", "cnn_study.py", "length_features.py")}


def _plain(value):
    return json.loads(json.dumps(value))


def _smoke_rows(rows, settings):
    groups = defaultdict(list)
    for row in rows:
        groups[collision_key(original_text(row))].append(row)
    rng = random.Random(settings.split_seed)
    candidates = list(groups.values())
    rng.shuffle(candidates)
    counts, selected = Counter(), []
    for group in candidates:
        if any(counts[row["label"]] < settings.smoke_rows // 2 for row in group):
            selected.extend(group)
            counts.update(row["label"] for row in group)
        if min(counts.get(0, 0), counts.get(1, 0)) >= settings.smoke_rows // 2:
            break
    return sorted(selected, key=lambda row: row["source_row"])


def _effective(config, settings):
    if not settings.smoke:
        return config
    return replace(config, embedding_dim=min(8, config.embedding_dim), num_filters=min(8, config.num_filters),
                   preprocessing=replace(config.preprocessing,
                                         max_vocabulary_size=min(512, config.preprocessing.max_vocabulary_size),
                                         max_sequence_length=min(64, config.preprocessing.max_sequence_length)))


def _settings(manifest):
    values = dict(manifest["settings"])
    values["training_seeds"] = tuple(values["training_seeds"])
    return StudySettings(**values)


def create_suite(run_dir, raw_dir=base.ROOT / "data", normalized_dir=None, settings=None,
                 configurations=None, preprocessing_configurations=None):
    """Prepare a train-only manifest. Never read validation or test here."""
    settings = settings or StudySettings()
    raw_dir = Path(raw_dir).resolve()
    if normalized_dir is None:
        normalized_dir = raw_dir / "normalized"
        if not (normalized_dir / "train.json").is_file():
            normalized_dir = base.ROOT / "cleaning" / "data" / "normalized"
    normalized_dir, run_dir = Path(normalized_dir).resolve(), Path(run_dir).resolve()
    datasets = load_datasets(raw_dir, normalized_dir, "train")
    if settings.smoke:
        datasets["raw"] = _smoke_rows(datasets["raw"], settings)
        ids = {row["source_row"] for row in datasets["raw"]}
        for name in ("normalized", "controlled"):
            datasets[name] = [row for row in datasets[name] if row["source_row"] in ids]
    configurations = list(configurations if configurations is not None else screening_configs())
    controls = list(preprocessing_configurations if preprocessing_configurations is not None else controlled_configs())
    if not configurations or configurations[0].config_id != "reference":
        raise ValueError("The first screening configuration must be the reference")
    if len({c.config_id for c in configurations + controls}) != len(configurations + controls):
        raise ValueError("Configuration IDs must be unique")
    folds = shared_folds(datasets["raw"], settings)
    if settings.smoke:
        folds = folds[:1]
    for rows in datasets.values():
        for fold in folds:
            partition_rows(rows, fold)
    group_labels = defaultdict(set)
    for row in datasets["raw"]:
        group_labels[collision_key(original_text(row))].add(row["label"])
    recipe = _plain({
        "format": "cnn_study.suite.v2", "mode": "smoke" if settings.smoke else "full",
        "settings": asdict(settings), "source_hashes": source_hashes(),
        "datasets": {"raw": str(raw_dir), "normalized": str(normalized_dir)},
        "training_sha256": {name: base.sha256(path / "train.json")
                            for name, path in (("raw", raw_dir), ("normalized", normalized_dir))},
        "screen_configs": [asdict(c) for c in configurations], "control_configs": [asdict(c) for c in controls],
        "source_rows": {name: [r["source_row"] for r in rows] for name, rows in datasets.items()},
        "folds": folds, "grouping": "casefold_words_only_v1",
        "mixed_label_collision_groups": sum(len(labels) > 1 for labels in group_labels.values()),
        "selection": "mean_macro_f1_at_0.5_then_std_then_parameters_then_id",
        "runtime_versions": base.runtime_info(torch.device("cpu")),
    })
    manifest = {**recipe, "run_signature": base.fingerprint(recipe)}
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    if path.exists() and base.read_json(path) != manifest:
        raise ValueError("Run directory has different data, settings, source or runtime; choose a new run name")
    base.write_json(path, manifest)
    return load_suite(run_dir)


def load_suite(run_dir, verify_sources=True):
    run_dir = Path(run_dir).resolve()
    manifest = base.read_json(run_dir / "manifest.json")
    unsigned = {key: value for key, value in manifest.items() if key != "run_signature"}
    if base.fingerprint(unsigned) != manifest["run_signature"]:
        raise ValueError("Study manifest was edited")
    if verify_sources and source_hashes() != manifest["source_hashes"]:
        raise ValueError("Study source changed; choose a new run name")
    for name, expected in manifest["training_sha256"].items():
        if base.sha256(Path(manifest["datasets"][name]) / "train.json") != expected:
            raise ValueError("Training data changed after study preparation")
    return {"run_dir": str(run_dir), "manifest": manifest}


def _training_data(manifest):
    datasets = load_datasets(manifest["datasets"]["raw"], manifest["datasets"]["normalized"], "train")
    for name, rows in datasets.items():
        ids = set(manifest["source_rows"][name])
        datasets[name] = [row for row in rows if row["source_row"] in ids]
    return datasets


def _require_training_runtime(run_dir, manifest, device):
    if base.runtime_info(torch.device("cpu")) != manifest["runtime_versions"]:
        raise ValueError("Training runtime changed; choose a new run name")
    path = run_dir / "execution_runtime.json"
    runtime = base.runtime_info(device)
    if path.exists() and base.read_json(path) != runtime:
        raise ValueError("Training device or runtime changed; choose a new run name")
    base.write_json(path, runtime)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_oom(error):
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def preflight(config, vocabulary, state, rows, device, seed=42):
    """Exercise Adam state allocation before choosing a microbatch size."""
    attempts = []
    for micro_batch in (32, 16, 8):
        model = optimizer = loader = None
        try:
            if device.type == "cuda":
                # Allocator statistics are unavailable until CUDA/HIP lazy init.
                _synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            base.seed_everything(seed)
            model = StudyCNN(vocabulary, config, state).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                         weight_decay=config.weight_decay)
            loader = make_loader(rows[:32], vocabulary, model, micro_batch)
            train_epoch(model, loader, optimizer, device)
            _synchronize(device)
            started = time.perf_counter()
            train_epoch(model, loader, optimizer, device)
            _synchronize(device)
            seconds = time.perf_counter() - started
            result = {"status": "complete", "micro_batch_size": micro_batch,
                      "effective_batch_size": 32, "seconds_per_effective_batch": seconds,
                      "examples_per_second": min(32, len(rows)) / seconds,
                      "parameter_count": sum(p.numel() for p in model.parameters()),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                      "attempts": attempts, "runtime": base.runtime_info(device)}
            return result
        except RuntimeError as error:
            attempts.append({"micro_batch_size": micro_batch, "error": str(error), "oom": _is_oom(error),
                             "traceback": traceback.format_exc()})
            if not _is_oom(error):
                break
        finally:
            del model, optimizer, loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {"status": "failed", "attempts": attempts}


def _train_fold(config, parts, settings, seed, device, progress):
    vocabulary = build_vocabulary(parts["train"], config.preprocessing)
    state = LengthState.fit(parts["train"], config.length_feature)
    check = preflight(config, vocabulary, state, parts["train"], device, seed)
    if check["status"] != "complete":
        return None, vocabulary, {"status": "failed", "preflight": check}
    # Preflight updates its own disposable model; reset every RNG before the fit.
    base.seed_everything(seed)
    model = StudyCNN(vocabulary, config, state).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    micro = check["micro_batch_size"]
    fit_loader = make_loader(parts["train"], vocabulary, model, micro, True, seed)
    stop_loader = make_loader(parts["early_stopping"], vocabulary, model, micro)
    best_loss, best_epoch, best_state, stale, history = math.inf, 0, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, (1 if settings.smoke else settings.max_epochs) + 1):
        training = train_epoch(model, fit_loader, optimizer, device)
        _, loss = score_loader(model, stop_loader, device)
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite early-stopping loss")
        history.append({"epoch": epoch, **training, "early_stopping_loss": loss})
        if loss < best_loss:
            best_loss, best_epoch, stale = loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    model.load_state_dict(best_state)
    scores, loss = predict_rows(model, parts["heldout"], vocabulary, device=device, batch_size=micro)
    tokens = [tokenize(row["prompt"], config.preprocessing) for row in parts["heldout"]]
    used = [token for values in tokens for token in values[:config.preprocessing.max_sequence_length]]
    result = {"status": "complete", "best_epoch": best_epoch, "history": history, "preflight": check,
              "training_seconds": time.perf_counter() - started, "vocabulary_size": len(vocabulary),
              "parameter_count": check["parameter_count"], "length_state": asdict(state),
              "heldout_unknown_token_rate": sum(t not in vocabulary for t in used) / len(used) if used else 0.0,
              "heldout_truncated_rows": sum(len(t) > config.preprocessing.max_sequence_length for t in tokens),
              "metrics": base.binary_metrics([r["label"] for r in parts["heldout"]], scores, loss=loss),
              "scores": scores.tolist()}
    return model, vocabulary, result


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rank(row):
    return (-row["macro_f1_mean"], row["macro_f1_std"], row["parameter_count"], row["config_id"])


def _run_recipe(run_dir, manifest, dataset, config, rows, seeds, device, progress):
    settings = _settings(manifest)
    config = _effective(config, settings)
    fingerprint_config = asdict(config)
    for key in ("config_id", "name"):
        fingerprint_config.pop(key)
    data_key = base.fingerprint([
        [row["source_row"], row["label"], tokenize(row["prompt"], config.preprocessing),
         original_text(row) if config.length_feature != "none" else None] for row in rows
    ])
    fold_results = []
    for seed in seeds:
        for fold in manifest["folds"]:
            identity = {"config": fingerprint_config, "data": data_key, "fold": fold,
                        "seed": seed, "run_signature": manifest["run_signature"],
                        "runtime": base.runtime_info(device)}
            key = base.fingerprint(identity)
            directory = run_dir / "_cache" / key
            completion = directory / "result.json"
            if completion.exists():
                result = base.read_json(completion)
                if result["identity"] != _plain(identity):
                    raise ValueError("Fold cache identity mismatch")
                if result["status"] == "complete":
                    base._verify_artifact(directory / "model.pt", result["checkpoint_sha256"])
                    base._verify_artifact(directory / "predictions.jsonl", result["predictions_sha256"])
                if progress:
                    progress(f"Resume {dataset}/{config.config_id}, seed {seed}, fold {fold['fold']}")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                parts = partition_rows(rows, fold)
                if progress:
                    progress(f"Train {dataset}/{config.config_id}, seed {seed}, fold {fold['fold']} on {device}")
                model, vocabulary, result = _train_fold(config, parts, settings, seed, device, progress)
                if result["status"] == "complete":
                    scores = result.pop("scores")
                    save_checkpoint(directory / "model.pt", model, vocabulary,
                                    run_signature=manifest["run_signature"], seed=seed, fold=fold["fold"],
                                    best_epoch=result["best_epoch"], source_hashes=manifest["source_hashes"])
                    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as output:
                        for row, score in zip(parts["heldout"], scores, strict=True):
                            output.write(json.dumps({"source_row": row["source_row"], "label": row["label"],
                                                     "injection_score": score}) + "\n")
                    result.update(checkpoint_sha256=base.sha256(directory / "model.pt"),
                                  predictions_sha256=base.sha256(directory / "predictions.jsonl"))
                result.update(identity=identity, seed=seed, fold=fold["fold"],
                              directory=str(directory.relative_to(run_dir)))
                base.write_json(completion, result)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            fold_results.append(result)
    complete = all(result["status"] == "complete" for result in fold_results)
    entry = {"dataset": dataset, "config_id": config.config_id, "config": asdict(config),
             "status": "complete" if complete else "failed", "fold_results": fold_results}
    if complete:
        values = [r["metrics"]["macro_f1"] for r in fold_results]
        entry.update(macro_f1_mean=float(np.mean(values)), macro_f1_std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                     parameter_count=float(np.mean([r["parameter_count"] for r in fold_results])))
    return entry


def _save_stage(run_dir, stage, manifest, entries, device):
    directory = run_dir / stage
    directory.mkdir(parents=True, exist_ok=True)
    base.write_json(directory / "results.json", {"run_signature": manifest["run_signature"],
                                                "runtime": base.runtime_info(device), "entries": entries})
    leaderboard = [{"dataset": e["dataset"], "config_id": e["config_id"],
                    "macro_f1_mean": e["macro_f1_mean"], "macro_f1_std": e["macro_f1_std"],
                    "parameter_count": e["parameter_count"], "fold_runs": len(e["fold_results"])}
                   for e in entries if e["status"] == "complete"]
    leaderboard.sort(key=lambda e: (e["dataset"], *_rank(e)))
    _write_csv(directory / "leaderboard.csv", leaderboard)
    return entries


def _load_stage(run_dir, stage, manifest):
    saved = base.read_json(run_dir / stage / "results.json")
    if saved["run_signature"] != manifest["run_signature"]:
        raise ValueError("Stage belongs to a different study")
    for entry in saved["entries"]:
        for result in entry["fold_results"]:
            if result["status"] == "complete":
                directory = run_dir / result["directory"]
                base._verify_artifact(directory / "model.pt", result["checkpoint_sha256"])
                base._verify_artifact(directory / "predictions.jsonl", result["predictions_sha256"])
    return saved["entries"]


def run_screening(run_dir, device="auto", progress=print):
    suite = load_suite(run_dir)
    manifest, run_dir = suite["manifest"], Path(suite["run_dir"])
    datasets, device = _training_data(manifest), base.resolve_device(device)
    _require_training_runtime(run_dir, manifest, device)
    entries = []
    for dataset in ("raw", "normalized", "controlled"):
        configurations = manifest["control_configs"] if dataset == "controlled" else manifest["screen_configs"]
        for value in configurations:
            entries.append(_run_recipe(run_dir, manifest, dataset, config_from_dict(value),
                                       datasets[dataset], (42,), device, progress))
    return _save_stage(run_dir, "screening", manifest, entries, device)


def _best(entries):
    entries = [entry for entry in entries if entry["status"] == "complete"]
    if not entries:
        raise ValueError("No complete eligible configuration")
    return min(entries, key=_rank)


def combined_config(entries, dataset):
    own = {e["config_id"]: e for e in entries if e["dataset"] == dataset and e["status"] == "complete"}
    if "reference" not in own:
        raise ValueError(f"The {dataset} reference must complete before combining candidates")
    reference = own["reference"]
    config = config_from_dict(reference["config"])
    score = reference["macro_f1_mean"]

    def improvement(names):
        candidates = [own[name] for name in names if name in own and own[name]["macro_f1_mean"] > score]
        return config_from_dict(_best(candidates)["config"]) if candidates else config

    vocabulary = improvement(("vocabulary_40k",)).preprocessing.max_vocabulary_size
    embedding = improvement(("embedding_128",)).embedding_dim
    window = improvement(("window_2048",)).preprocessing.max_sequence_length
    length = improvement(("length_tokens", "length_characters")).length_feature
    architecture = improvement(("filters_512", "depth_2", "depth_3", "kernel_7", "parallel_357"))
    controls = [e for e in entries if e["dataset"] == "controlled" and e["status"] == "complete"
                and (dataset == "raw" or e["config"]["preprocessing"]["lowercase"])]
    preprocessing = config_from_dict(_best(controls)["config"]).preprocessing if controls else config.preprocessing
    preprocessing = replace(preprocessing, max_vocabulary_size=vocabulary, max_sequence_length=window)
    return replace(architecture, config_id="combined", name=f"Combined improvements ({dataset})",
                   preprocessing=preprocessing, embedding_dim=embedding, length_feature=length)


def run_combined(run_dir, device="auto", progress=print):
    suite = load_suite(run_dir)
    manifest, run_dir = suite["manifest"], Path(suite["run_dir"])
    screen = _load_stage(run_dir, "screening", manifest)
    datasets, device = _training_data(manifest), base.resolve_device(device)
    _require_training_runtime(run_dir, manifest, device)
    entries = [_run_recipe(run_dir, manifest, dataset, combined_config(screen, dataset), datasets[dataset],
                           (42,), device, progress) for dataset in ("raw", "normalized")]
    return _save_stage(run_dir, "combined", manifest, entries, device)


def run_confirmation(run_dir, device="auto", progress=print):
    suite = load_suite(run_dir)
    manifest, run_dir = suite["manifest"], Path(suite["run_dir"])
    candidates = _load_stage(run_dir, "screening", manifest) + _load_stage(run_dir, "combined", manifest)
    datasets, device = _training_data(manifest), base.resolve_device(device)
    _require_training_runtime(run_dir, manifest, device)
    entries = []
    for dataset in ("raw", "normalized"):
        own = [e for e in candidates if e["dataset"] == dataset]
        reference = next(e for e in own if e["config_id"] == "reference")
        selected = _best(own)
        for entry in {e["config_id"]: e for e in (reference, selected)}.values():
            entries.append(_run_recipe(run_dir, manifest, dataset, config_from_dict(entry["config"]),
                                       datasets[dataset], _settings(manifest).training_seeds, device, progress))
    return _save_stage(run_dir, "confirmation", manifest, entries, device)


def run_length_baselines(run_dir):
    """Logistic references use the same training/held-out rows as the CNN."""
    suite = load_suite(run_dir)
    manifest, run_dir = suite["manifest"], Path(suite["run_dir"])
    datasets, entries = _training_data(manifest), []
    for dataset in ("raw", "normalized"):
        for mode in ("log_tokens", "log_characters"):
            results = []
            for fold in manifest["folds"]:
                parts = partition_rows(datasets[dataset], fold)
                state = LengthState.fit(parts["train"], mode)
                classifier = LogisticRegression(max_iter=1000, random_state=42)
                classifier.fit(state.transform(parts["train"]).numpy(), [r["label"] for r in parts["train"]])
                scores = classifier.predict_proba(state.transform(parts["heldout"]).numpy())[:, 1]
                results.append({"fold": fold["fold"], "length_state": asdict(state),
                                "coefficient": float(classifier.coef_[0, 0]), "intercept": float(classifier.intercept_[0]),
                                "source_rows": [r["source_row"] for r in parts["heldout"]],
                                "scores": scores.tolist(),
                                "metrics": base.binary_metrics([r["label"] for r in parts["heldout"]], scores)})
            entries.append({"dataset": dataset, "feature": mode, "fold_results": results,
                            "macro_f1_mean": float(np.mean([r["metrics"]["macro_f1"] for r in results]))})
    directory = run_dir / "length_baselines"
    directory.mkdir(parents=True, exist_ok=True)
    base.write_json(directory / "results.json", {"run_signature": manifest["run_signature"], "entries": entries})
    return entries


def _final_fit(run_dir, manifest, dataset, entry, rows, device, progress):
    config = config_from_dict(entry["config"])
    directory = run_dir / "refit" / dataset / config.config_id
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "result.json"
    epochs = max(1, int(np.median([r["best_epoch"] for r in entry["fold_results"]])))
    if metadata_path.exists():
        result = base.read_json(metadata_path)
        if result["run_signature"] != manifest["run_signature"] or result["epochs"] != epochs:
            raise ValueError("Final checkpoint identity changed")
        base._verify_artifact(directory / "model.pt", result["checkpoint_sha256"])
    else:
        vocabulary = build_vocabulary(rows, config.preprocessing)
        state = LengthState.fit(rows, config.length_feature)
        check = preflight(config, vocabulary, state, rows, device, 42)
        if check["status"] != "complete":
            raise RuntimeError(f"Final model cannot run on requested device: {check}")
        if progress:
            progress(f"Refit {dataset}/{config.config_id}: {len(rows)} rows, {epochs} epochs")
        base.seed_everything(42)
        model = StudyCNN(vocabulary, config, state).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        loader = make_loader(rows, vocabulary, model, check["micro_batch_size"], True, 42)
        history = [dict(epoch=epoch, **train_epoch(model, loader, optimizer, device))
                   for epoch in range(1, epochs + 1)]
        save_checkpoint(directory / "model.pt", model, vocabulary, run_signature=manifest["run_signature"],
                        mode=manifest["mode"], dataset=dataset, epochs=epochs, seed=42,
                        source_hashes=manifest["source_hashes"])
        result = {"run_signature": manifest["run_signature"], "config": asdict(config), "epochs": epochs,
                  "seed": 42, "rows": len(rows), "length_state": asdict(state), "history": history,
                  "preflight": check, "checkpoint_sha256": base.sha256(directory / "model.pt")}
        base.write_json(metadata_path, result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"kind": "cnn", "path": str((directory / "model.pt").relative_to(run_dir)),
            "sha256": result["checkpoint_sha256"], "config_id": config.config_id,
            "micro_batch_size": result["preflight"]["micro_batch_size"]}


def refit_suite(run_dir, device="auto", progress=print):
    """Retrain the reference and confirmation winner; validation stays unopened."""
    suite = load_suite(run_dir)
    manifest, run_dir = suite["manifest"], Path(suite["run_dir"])
    confirmation = _load_stage(run_dir, "confirmation", manifest)
    datasets, device = _training_data(manifest), base.resolve_device(device)
    _require_training_runtime(run_dir, manifest, device)
    models, selected = {}, {}
    for dataset in ("raw", "normalized"):
        own = [entry for entry in confirmation if entry["dataset"] == dataset]
        winner = _best(own)
        reference = next(entry for entry in own if entry["config_id"] == "reference")
        if reference["status"] != "complete":
            raise ValueError(f"The {dataset} reference failed confirmation")
        selected[dataset] = {key: winner[key] for key in ("config_id", "macro_f1_mean", "macro_f1_std", "parameter_count")}
        for role, entry in (("selected", winner), ("reference", reference)):
            models[f"{dataset}_{role}"] = {"dataset": dataset,
                **_final_fit(run_dir, manifest, dataset, entry, datasets[dataset], device, progress)}
        for mode in ("log_tokens", "log_characters"):
            rows = datasets[dataset]
            state = LengthState.fit(rows, mode)
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(state.transform(rows).numpy(), [r["label"] for r in rows])
            path = run_dir / "refit" / dataset / f"{mode}.json"
            base.write_json(path, {"length_state": asdict(state), "coefficient": float(classifier.coef_[0, 0]),
                                   "intercept": float(classifier.intercept_[0]), "run_signature": manifest["run_signature"]})
            models[f"{dataset}_{mode}"] = {"dataset": dataset, "kind": "length",
                                          "path": str(path.relative_to(run_dir)), "sha256": base.sha256(path)}
    record = {"run_signature": manifest["run_signature"], "mode": manifest["mode"], "models": models,
              "selected": selected, "overall_selected_dataset": min(selected, key=lambda d: (
                  -selected[d]["macro_f1_mean"], selected[d]["macro_f1_std"], selected[d]["parameter_count"], d))}
    base.write_json(run_dir / "refit" / "selected.json", record)
    return record


def _score_spec(run_dir, spec, rows, device):
    path = run_dir / spec["path"]
    base._verify_artifact(path, spec["sha256"])
    if spec["kind"] == "cnn":
        model, vocabulary, config, _ = base.load_checkpoint(path, device)
        scores, _ = predict_rows(model, rows, vocabulary, device=device,
                                 batch_size=spec.get("micro_batch_size", 32))
        return scores, model, vocabulary, config
    saved = base.read_json(path)
    values = LengthState(**saved["length_state"]).transform(rows).numpy()[:, 0]
    logits = saved["coefficient"] * values + saved["intercept"]
    scores = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    return scores, None, None, None


def _length_bands(rows, quartiles):
    counts = length_counts(rows, "log_tokens")
    return np.searchsorted(np.asarray(quartiles), counts, side="right")


def length_slice_report(rows, scores, quartiles, threshold=0.5):
    bands = _length_bands(rows, quartiles)
    result = []
    for band in range(4):
        indices = np.flatnonzero(bands == band)
        if not len(indices):
            continue
        labels = np.array([rows[i]["label"] for i in indices])
        predictions = np.asarray(scores)[indices] >= threshold
        benign, attacks = labels == 0, labels == 1
        result.append({"length_band": band + 1, "rows": len(indices), "benign_rows": int(benign.sum()),
                       "injection_rows": int(attacks.sum()),
                       "false_positive_rate": float(predictions[benign].mean()) if benign.any() else None,
                       "injection_recall": float(predictions[attacks].mean()) if attacks.any() else None})
    return result


def _balanced_length_mask(rows, eligible, quartiles):
    bands = _length_bands(rows, quartiles)
    labels = np.array([row["label"] for row in rows])
    selected = np.zeros(len(rows), dtype=bool)
    for band in range(4):
        indices = [np.flatnonzero(eligible & (bands == band) & (labels == label)) for label in (0, 1)]
        size = min(map(len, indices))
        for values in indices:
            selected[values[:size]] = True
    return selected


def _padding_report(model, vocabulary, config, rows, scores, device):
    suffix = " neutral context" * 8
    selected, modified, counts = [], [], Counter()
    for index, row in enumerate(rows):
        if counts[row["label"]] >= 100:
            continue
        if len(tokenize(row["prompt"] + suffix, config.preprocessing)) > config.preprocessing.max_sequence_length:
            continue
        selected.append(index)
        modified.append({**row, "prompt": row["prompt"] + suffix,
                         "original_prompt": original_text(row) + suffix})
        counts[row["label"]] += 1
    if not selected:
        return {"rows": 0, "interpretation": "No examples fit the window with the full original text and padding"}
    padded, _ = predict_rows(model, modified, vocabulary, device=device, batch_size=8)
    original = np.asarray(scores)[selected]
    return {"rows": len(selected), "by_label": dict(counts), "source_rows": [rows[i]["source_row"] for i in selected],
            "mean_absolute_score_change": float(np.abs(padded - original).mean()),
            "prediction_flips_at_0.5": int(((padded >= 0.5) != (original >= 0.5)).sum()),
            "original_scores": original.tolist(), "padded_scores": padded.tolist(),
            "interpretation": "Neutral suffix assumed label-preserving; original text remains fully visible. Review examples before interpreting."}


def _write_scored_rows(path, rows, scores):
    with path.open("w", encoding="utf-8") as output:
        for row, score in zip(rows, scores, strict=True):
            output.write(json.dumps({"row_index": row["row_index"], "source_row": row["source_row"],
                                     "label": row["label"], "injection_score": float(score)}) + "\n")


def analyze_validation(run_dir, device="auto"):
    """Calibrate only on common, training-disjoint validation records."""
    suite = load_suite(run_dir)
    manifest, run_dir = suite["manifest"], Path(suite["run_dir"])
    selected = base.read_json(run_dir / "refit" / "selected.json")
    if selected["run_signature"] != manifest["run_signature"]:
        raise ValueError("Refit belongs to another study")
    datasets = load_datasets(manifest["datasets"]["raw"], manifest["datasets"]["normalized"], "validation")
    training = load_datasets(manifest["datasets"]["raw"], manifest["datasets"]["normalized"], "train")
    training_keys = {collision_key(original_text(row)) for row in training["raw"]}
    common_ids = {row["source_row"] for row in datasets["raw"]} & {row["source_row"] for row in datasets["normalized"]}
    eligible_ids = {row["source_row"] for row in datasets["raw"] if row["source_row"] in common_ids
                    and collision_key(original_text(row)) not in training_keys}
    quartiles = np.quantile(length_counts(training["raw"], "log_tokens"), [0.25, 0.5, 0.75]).tolist()
    directory = run_dir / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    reports, profiles, predictions, robustness, slices, coverage, input_diagnostics = [], {}, {}, {}, {}, {}, {}
    device = base.resolve_device(device)
    for name, spec in selected["models"].items():
        rows = datasets[spec["dataset"]]
        mask = np.array([row["source_row"] in eligible_ids for row in rows])
        labels = np.array([row["label"] for row in rows])
        if set(labels[mask]) != {0, 1}:
            raise ValueError("Common training-disjoint validation is missing a class")
        scores, model, vocabulary, config = _score_spec(run_dir, spec, rows, device)
        profiles[name] = base.threshold_profiles(labels[mask], scores[mask])
        for split, selected_mask in (("validation_full_development", np.ones(len(rows), dtype=bool)),
                                      ("validation_common_disjoint_development", mask)):
            reports.extend(base._profile_rows(name, split, rows, scores, profiles[name], selected_mask))
        balanced = _balanced_length_mask(rows, mask, quartiles)
        if balanced.any():
            reports.extend(base._profile_rows(name, "validation_length_matched_development", rows, scores,
                                              profiles[name], balanced))
        slices[name] = length_slice_report([r for r, keep in zip(rows, mask) if keep], scores[mask], quartiles)
        coverage[name] = {"full_rows": len(rows), "common_disjoint_rows": int(mask.sum()),
                          "length_matched_rows": int(balanced.sum())}
        if model is not None:
            diagnostic_rows = []
            for row, score in zip(rows, scores, strict=True):
                tokens = tokenize(row["prompt"], config.preprocessing)
                used = tokens[:config.preprocessing.max_sequence_length]
                prediction = int(score >= 0.5)
                diagnostic_rows.append({"source_row": row["source_row"], "row_index": row["row_index"],
                    "label": row["label"], "prediction": prediction, "injection_score": float(score),
                    "original_characters": len(original_text(row)),
                    "original_tokens": int(length_counts([row], "log_tokens")[0]),
                    "model_tokens": len(tokens), "truncated": len(tokens) > config.preprocessing.max_sequence_length,
                    "unknown_token_rate": sum(t not in vocabulary for t in used) / len(used) if used else 0.0,
                    "error_type": "false_positive" if prediction and not row["label"] else
                                  "false_negative" if row["label"] and not prediction else "correct"})
            diagnostic_path = directory / f"{name}_validation_diagnostics.jsonl"
            with diagnostic_path.open("w", encoding="utf-8") as output:
                for row in diagnostic_rows:
                    output.write(json.dumps(row) + "\n")
            input_diagnostics[name] = {"path": str(diagnostic_path.relative_to(run_dir)),
                                       "truncated_rows": sum(r["truncated"] for r in diagnostic_rows),
                                       "mean_unknown_token_rate": float(np.mean([r["unknown_token_rate"] for r in diagnostic_rows]))}
        if model is not None and name.endswith("_selected"):
            clean_rows = [row for row, keep in zip(rows, mask) if keep]
            robustness[name] = _padding_report(model, vocabulary, config, clean_rows, scores[mask], device)
        path = directory / f"{name}_validation_predictions.jsonl"
        _write_scored_rows(path, rows, scores)
        predictions[name] = {"path": str(path.relative_to(run_dir)), "sha256": base.sha256(path)}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metadata = {"run_signature": manifest["run_signature"], "mode": manifest["mode"],
                "source_hashes": source_hashes(), "models": selected["models"], "profiles": profiles,
                "validation_sha256": {name: base.sha256(Path(path) / "validation.json")
                                      for name, path in manifest["datasets"].items()},
                "calibration_source_rows": sorted(eligible_ids), "training_length_quartiles": quartiles,
                "training_collision_keys": sorted(training_keys),
                "interpretation": "Validation results are development results; test is a previously inspected benchmark."}
    report = {"metadata": metadata, "reports": reports, "coverage": coverage, "length_slices": slices,
              "neutral_padding": robustness, "predictions": predictions, "input_diagnostics": input_diagnostics}
    frozen_path = directory / "frozen_evaluation.json"
    if frozen_path.exists() and base.read_json(frozen_path)["metadata"] != _plain(metadata):
        raise ValueError("Analysis differs from frozen evaluation; use a new run name")
    base.write_json(directory / "validation_report.json", report)
    return report


def freeze_study(run_dir):
    suite = load_suite(run_dir)
    run_dir = Path(suite["run_dir"])
    report = base.read_json(run_dir / "analysis" / "validation_report.json")
    metadata = report["metadata"]
    if metadata["run_signature"] != suite["manifest"]["run_signature"] or metadata["source_hashes"] != source_hashes():
        raise ValueError("Validation analysis no longer matches the study")
    for spec in metadata["models"].values():
        base._verify_artifact(run_dir / spec["path"], spec["sha256"])
    for spec in report["predictions"].values():
        base._verify_artifact(run_dir / spec["path"], spec["sha256"])
    for name, location in suite["manifest"]["datasets"].items():
        if base.sha256(Path(location) / "validation.json") != metadata["validation_sha256"][name]:
            raise ValueError("Validation data changed after calibration")
    frozen = {"metadata": metadata, "freeze_signature": base.fingerprint(metadata)}
    path = run_dir / "analysis" / "frozen_evaluation.json"
    if path.exists() and base.read_json(path) != frozen:
        raise ValueError("Frozen evaluation was edited")
    base.write_json(path, frozen)
    return path


def evaluate_study_test(run_dir, device="auto"):
    """Read test only after a non-smoke study has frozen its models/thresholds."""
    suite = load_suite(run_dir)
    manifest, run_dir = suite["manifest"], Path(suite["run_dir"])
    if manifest["mode"] != "full":
        raise ValueError("Smoke studies cannot evaluate the test benchmark")
    frozen = base.read_json(run_dir / "analysis" / "frozen_evaluation.json")
    metadata = frozen["metadata"]
    if base.fingerprint(metadata) != frozen["freeze_signature"]:
        raise ValueError("Frozen evaluation was edited")
    if metadata["run_signature"] != manifest["run_signature"] or metadata["source_hashes"] != source_hashes():
        raise ValueError("Frozen study identity changed")
    for name, path in manifest["datasets"].items():
        if base.sha256(Path(path) / "validation.json") != metadata["validation_sha256"][name]:
            raise ValueError("Validation data changed after freezing")
    for spec in metadata["models"].values():
        base._verify_artifact(run_dir / spec["path"], spec["sha256"])
    identity = {"freeze_signature": frozen["freeze_signature"],
                "test_sha256": {name: base.sha256(Path(path) / "test.json") for name, path in manifest["datasets"].items()}}
    directory = run_dir / "analysis" / ("test_" + base.fingerprint(identity)[:12])
    path = directory / "report.json"
    if path.exists():
        report = base.read_json(path)
        if report["identity"] != identity:
            raise ValueError("Test cache identity mismatch")
        for spec in report["predictions"].values():
            base._verify_artifact(run_dir / spec["path"], spec["sha256"])
        return report
    datasets = load_datasets(manifest["datasets"]["raw"], manifest["datasets"]["normalized"], "test")
    common = {r["source_row"] for r in datasets["raw"]} & {r["source_row"] for r in datasets["normalized"]}
    training_keys = set(metadata["training_collision_keys"])
    eligible = {r["source_row"] for r in datasets["raw"] if r["source_row"] in common
                and collision_key(original_text(r)) not in training_keys}
    reports, predictions, slices = [], {}, {}
    directory.mkdir(parents=True, exist_ok=True)
    device = base.resolve_device(device)
    for name, spec in metadata["models"].items():
        rows = datasets[spec["dataset"]]
        mask = np.array([r["source_row"] in eligible for r in rows])
        if {r["label"] for r, keep in zip(rows, mask) if keep} != {0, 1}:
            raise ValueError("Common eligible test records are missing a class")
        scores, model, _, _ = _score_spec(run_dir, spec, rows, device)
        for split, subset in (("test_full_known_benchmark", np.ones(len(rows), dtype=bool)),
                               ("test_common_disjoint_known_benchmark", mask)):
            reports.extend(base._profile_rows(name, split, rows, scores, metadata["profiles"][name], subset))
        slices[name] = length_slice_report([r for r, keep in zip(rows, mask) if keep], scores[mask],
                                           metadata["training_length_quartiles"])
        prediction_path = directory / f"{name}_predictions.jsonl"
        _write_scored_rows(prediction_path, rows, scores)
        predictions[name] = {"path": str(prediction_path.relative_to(run_dir)), "sha256": base.sha256(prediction_path)}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {"identity": identity, "reports": reports, "predictions": predictions, "length_slices": slices,
              "interpretation": "Previously inspected benchmark; no test-based model or threshold selection."}
    base.write_json(path, report)
    return report
