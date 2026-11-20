"""Fábrica dos cinco modelos candidatos e amostragem de hiperparâmetros a partir do YAML."""

from __future__ import annotations

import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = ("logreg", "random_forest", "xgboost", "lightgbm", "catboost")


def suggest_params(trial: optuna.Trial, space: dict, prefix: str = "") -> dict:
    """Sugere os hiperparâmetros de 'space["search"]'. O prefixo evita colisão de nomes
    quando vários modelos compartilham o mesmo trial (comitê em A2)."""
    params = {}
    for name, spec in space.get("search", {}).items():
        key = f"{prefix}{name}"
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(key, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "float":
            params[name] = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(key, spec["choices"])
        else:
            raise ValueError(f"Tipo de hiperparâmetro desconhecido: {kind}")
    return params


def build_model(
    name: str,
    params: dict,
    space: dict,
    seed: int,
    n_jobs: int,
    categorical: list[str],
    numeric: list[str],
):
    """Instancia o modelo com hiperparâmetros fixos (YAML) + sugeridos (params)."""
    p = {**space.get("fixed", {}), **params}
    if name == "logreg":
        # Único modelo sensível à escala e a valores ausentes: imputa (mediana + indicador
        # de ausência), padroniza numéricas e codifica categóricas. Os modelos de árvore
        # tratam ausentes nativamente. Em Taiwan não há ausentes (imputação inócua).
        num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                        ("sc", StandardScaler())])
        cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore"))])
        pre = ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(random_state=seed, **p))])
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **p)
    if name == "xgboost":
        return XGBClassifier(random_state=seed, n_jobs=n_jobs, **p)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=n_jobs, **p)
    if name == "catboost":
        return CatBoostClassifier(random_seed=seed, thread_count=n_jobs, **p)
    raise ValueError(f"Modelo desconhecido: {name}")


def predict_scores(model, X) -> np.ndarray:
    """Probabilidade estimada da classe positiva (inadimplência)."""
    return np.asarray(model.predict_proba(X))[:, 1]
