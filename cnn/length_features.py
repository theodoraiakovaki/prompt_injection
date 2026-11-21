"""Versioned text processing and training-fitted document length features."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PreprocessingConfig:
    lowercase: bool = True
    punctuation_mode: str = "keep"
    max_vocabulary_size: int = 20_000
    max_sequence_length: int = 1_024

    def __post_init__(self):
        if type(self.lowercase) is not bool:
            raise ValueError("lowercase must be boolean")
        if self.punctuation_mode not in {"keep", "remove_bang_question", "words_only"}:
            raise ValueError("Unknown punctuation mode")
        if self.max_vocabulary_size < 2 or self.max_sequence_length < 1:
            raise ValueError("Invalid vocabulary or sequence limit")


def tokenize(text, config=PreprocessingConfig()):
    text = text.lower() if config.lowercase else text
    if config.punctuation_mode == "words_only":
        return re.findall(r"\w+", text)
    tokens = re.findall(r"\w+|[^\w\s]", text)
    if config.punctuation_mode == "remove_bang_question":
        tokens = [token for token in tokens if token not in {"!", "?"}]
    return tokens


def original_text(row):
    return row.get("original_prompt", row["prompt"])


def length_counts(rows, mode):
    if mode not in {"log_tokens", "log_characters"}:
        raise ValueError("Unknown length feature")
    # Independent of the experiment's case, punctuation, vocabulary and window.
    return np.array([
        len(re.findall(r"\w+|[^\w\s]", original_text(row)))
        if mode == "log_tokens" else len(original_text(row))
        for row in rows
    ], dtype=np.float64)


@dataclass(frozen=True)
class LengthState:
    mode: str = "none"
    mean: float = 0.0
    scale: float = 1.0
    version: str = "original_regex_log1p_v1"

    def __post_init__(self):
        if self.mode not in {"none", "log_tokens", "log_characters"}:
            raise ValueError("Unknown length feature")
        if self.version != "original_regex_log1p_v1":
            raise ValueError("Unsupported length transformation version")
        if not math.isfinite(self.mean) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Length statistics must be finite with positive scale")

    @classmethod
    def fit(cls, fit_rows, mode):
        if mode == "none":
            return cls()
        if not fit_rows:
            raise ValueError("Cannot fit length statistics without training rows")
        values = np.log1p(length_counts(fit_rows, mode))
        return cls(mode, float(values.mean()), float(values.std()) or 1.0)

    def transform(self, rows):
        if self.mode == "none":
            return torch.empty((len(rows), 0), dtype=torch.float32)
        values = (np.log1p(length_counts(rows, self.mode)) - self.mean) / self.scale
        return torch.tensor(values[:, None], dtype=torch.float32)
