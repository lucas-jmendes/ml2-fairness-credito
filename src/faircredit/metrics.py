"""Métricas de desempenho e de justiça.

Convenção: classe positiva = 1 = desfecho adverso (inadimplência em Taiwan, negação na HMDA);
predição positiva = prevê o desfecho adverso. Regra de decisão: score >= limiar.

Implementações próprias em NumPy (rápidas, usadas dentro dos estudos do Optuna);
tests/test_metrics.py confere os valores contra o Fairlearn e o scikit-learn.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    pos = y_true == 1
    tpr = y_pred[pos].mean() if pos.any() else np.nan
    fpr = y_pred[~pos].mean() if (~pos).any() else np.nan
    return float(tpr), float(fpr)


def _valid(groups: np.ndarray) -> np.ndarray:
    """Máscara das linhas com grupo definido (None = fora da métrica deste atributo)."""
    return np.array([g is not None for g in groups], dtype=bool)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tpr, fpr = _rates(y_true, y_pred)
    return (tpr + (1.0 - fpr)) / 2.0


def group_rates(y_true, y_pred, groups) -> dict[str, dict]:
    """TPR, FPR, taxa de seleção (predições positivas), taxa-base e n de cada grupo."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups, dtype=object)
    m0 = _valid(groups)
    out = {}
    for g in sorted(set(groups[m0])):
        m = m0 & (groups == g)
        tpr, fpr = _rates(y_true[m], y_pred[m])
        out[str(g)] = {
            "tpr": tpr, "fpr": fpr, "selection_rate": float(y_pred[m].mean()),
            "base_rate": float(y_true[m].mean()), "n": int(m.sum()),
        }
    return out


def fairness_metrics(y_true, y_pred, groups) -> dict[str, float]:
    """Violação de Equalized Odds (maior diferença entre grupos na TPR ou na FPR; igual ao
    fairlearn equalized_odds_difference com agg='worst_case'), diferenças de TPR/FPR,
    diferença de paridade demográfica e as taxas de cada grupo."""
    r = group_rates(y_true, y_pred, groups)
    tprs = [v["tpr"] for v in r.values()]
    fprs = [v["fpr"] for v in r.values()]
    sel = [v["selection_rate"] for v in r.values()]
    out = {
        "eo_violation": float(max(max(tprs) - min(tprs), max(fprs) - min(fprs))),
        "tpr_diff": float(max(tprs) - min(tprs)),
        "fpr_diff": float(max(fprs) - min(fprs)),
        "dp_difference": float(max(sel) - min(sel)),
    }
    for g, v in r.items():
        for k in ("tpr", "fpr", "selection_rate"):
            out[f"{k}_{g}"] = v[k]
    return out


def equalized_odds_violation(y_true, y_pred, groups) -> float:
    return fairness_metrics(y_true, y_pred, groups)["eo_violation"]


def demographic_parity_difference(y_pred, groups) -> float:
    """Maior diferença entre grupos na taxa de predições positivas."""
    y_pred, groups = np.asarray(y_pred), np.asarray(groups, dtype=object)
    m0 = _valid(groups)
    rates = [y_pred[m0 & (groups == g)].mean() for g in set(groups[m0])]
    return float(max(rates) - min(rates))


def best_threshold_balanced_accuracy(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Limiar que maximiza a acurácia balanceada (busca exata sobre a curva ROC).

    Devolve (limiar, acurácia balanceada). Ignora o limiar infinito (ninguém positivo).
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    ba = (tpr + 1.0 - fpr) / 2.0
    i = int(np.argmax(ba[1:])) + 1
    return float(thr[i]), float(ba[i])


def evaluate(y_true, scores, groups: dict[str, np.ndarray], threshold: float) -> dict[str, float]:
    """Desempenho + justiça para cada atributo sensível (chaves prefixadas pelo atributo,
    ex.: 'sex_eo_violation', 'age_tpr_lt25')."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    out = {
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float((y_pred == y_true).mean()),
        "positive_rate": float(y_pred.mean()),
    }
    for attr, g in groups.items():
        for k, v in fairness_metrics(y_true, y_pred, g).items():
            out[f"{attr}_{k}"] = v
    return out
