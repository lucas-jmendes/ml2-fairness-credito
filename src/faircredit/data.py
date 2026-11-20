"""Carga das bases e construção dos grupos sensíveis.

Cada base pode ter vários atributos sensíveis (ex.: sexo e idade < 25 em Taiwan; raça e sexo
na HMDA). Eles são devolvidos separados das variáveis de entrada e nunca fazem parte de X.
Linhas cujo valor não está no mapeamento de um atributo recebem None e ficam fora das
métricas daquele atributo (ex.: "Joint" no sexo da HMDA).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from faircredit.config import resolve


@dataclass
class Dataset:
    X: pd.DataFrame                  # variáveis de entrada (sem atributos sensíveis)
    y: np.ndarray                    # rótulo binário (1 = desfecho adverso: inadimplência/negação)
    groups: dict[str, np.ndarray]    # atributo sensível -> grupo de cada linha (None = fora)
    strata: np.ndarray               # chave de estratificação da divisão (rótulo x atributos)
    categorical: list[str]
    numeric: list[str]
    name: str


def _groups(col: pd.Series, spec: dict) -> np.ndarray:
    if "map" in spec:
        g = col.map(spec["map"])
    elif "below" in spec:
        low, high = spec["labels"]
        g = pd.Series(np.where(col < spec["below"], low, high), index=col.index)
    else:
        raise ValueError(f"Especificação de atributo sensível inválida: {spec}")
    return g.astype(object).where(g.notna(), None).to_numpy()


def _read_taiwan(d: dict):
    path = resolve(d["raw_dir"]) / d["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode antes: python scripts/download_taiwan.py")
    df = pd.read_excel(path, header=d["header_row"], engine="calamine")
    y = df[d["target"]].astype(int).to_numpy()
    for col, mapping in (d.get("recode") or {}).items():
        df[col] = df[col].replace(mapping)
    removed = {d["target"], *d["drop_columns"], *d["exclude_from_features"]}
    features = [c for c in df.columns if c not in removed]
    return df, y, features


def _read_hmda(d: dict):
    path = resolve(d["processed_file"])
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode antes: python scripts/download_hmda.py")
    df = pd.read_csv(path)
    return df, df["y"].astype(int).to_numpy(), list(d["features"])


READERS = {"taiwan": _read_taiwan, "hmda": _read_hmda}


def load_dataset(cfg: dict) -> Dataset:
    d = cfg["dataset"]
    df, y, features = READERS[d["loader"]](d)

    groups = {name: _groups(df[spec["column"]], spec) for name, spec in d["sensitive"].items()}

    # Garantia do protocolo: nenhum atributo sensível entra como variável de entrada.
    forbidden = {spec["column"] for spec in d["sensitive"].values()} | set(d.get("exclude_from_features", []))
    leaked = forbidden & set(features)
    if leaked:
        raise AssertionError(f"Atributos sensíveis nas entradas: {leaked}")

    strata = pd.Series(y).astype(str)
    for name in d["stratify_by"]:
        strata = strata + "|" + pd.Series(groups[name]).fillna("NA").astype(str)

    X = df[features].copy()
    categorical = [c for c in d["categorical"] if c in features]
    numeric = [c for c in features if c not in categorical]
    return Dataset(X, y, groups, strata.to_numpy(), categorical, numeric, d["name"])
