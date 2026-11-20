"""Divisão treino/validação/teste estratificada (rótulo x atributos sensíveis)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    strata: np.ndarray,
    seed: int,
    train: float = 0.6,
    val: float = 0.2,
    test: float = 0.2,
) -> dict[str, np.ndarray]:
    """Índices (ordenados) de cada partição. A mesma semente gera sempre a mesma divisão."""
    if not np.isclose(train + val + test, 1.0):
        raise ValueError("As proporções da divisão devem somar 1.")
    strata = np.asarray(strata)
    idx = np.arange(len(strata))
    idx_trval, idx_test = train_test_split(idx, test_size=test, stratify=strata, random_state=seed)
    idx_train, idx_val = train_test_split(
        idx_trval, test_size=val / (train + val), stratify=strata[idx_trval], random_state=seed
    )
    return {"train": np.sort(idx_train), "val": np.sort(idx_val), "test": np.sort(idx_test)}


def split_hash(splits: dict[str, np.ndarray]) -> str:
    """Impressão digital da divisão (para não retomar um estudo sobre outra divisão)."""
    h = hashlib.sha256()
    for name in ("train", "val", "test"):
        h.update(name.encode())
        h.update(np.asarray(splits[name], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def split_summary(y: np.ndarray, groups: dict[str, np.ndarray], splits: dict[str, np.ndarray]) -> pd.DataFrame:
    """Tamanho, participação e taxa do desfecho por grupo de cada atributo, em cada partição."""
    rows = []
    for part, idx in splits.items():
        for attr, g in groups.items():
            gi = g[idx]
            for value in sorted({v for v in gi if v is not None}):
                m = gi == value
                rows.append({
                    "split": part, "attribute": attr, "group": value, "n": int(m.sum()),
                    "share_of_split": m.mean(), "positive_rate": y[idx][m].mean(),
                })
    return pd.DataFrame(rows)
