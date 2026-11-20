"""Braço A0: cada um dos cinco modelos ajustado só por desempenho, sem mitigação.

Para cada modelo, um estudo do Optuna (objetivo único) maximiza a acurácia balanceada na
validação. Em cada trial o limiar de decisão é o que maximiza a acurácia balanceada na
validação (busca exata na curva ROC), então A0 não desperdiça orçamento procurando o limiar.
As métricas de justiça de todos os atributos sensíveis são registradas em todos os trials,
mas não orientam a busca. O melhor trial é retreinado no treino (mesma semente) e avaliado
uma única vez no teste. Os atributos sensíveis só são usados para calcular métricas.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
import pandas as pd

from faircredit.config import resolve
from faircredit.data import load_dataset
from faircredit.metrics import best_threshold_balanced_accuracy, evaluate
from faircredit.models import build_model, predict_scores, suggest_params
from faircredit.split import split_hash, split_summary, stratified_split
from faircredit.tuning import run_study

ARM = "a0"


def _prepare(cfg: dict, seed: int):
    data = load_dataset(cfg)
    sp = cfg["split"]
    splits = stratified_split(data.strata, seed, sp["train"], sp["val"], sp["test"])
    return data, splits


def _data_hash(data) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(data.X, index=False).to_numpy().tobytes())
    h.update(np.asarray(data.y, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _fingerprint(data, splits, space: dict) -> str:
    """Divisão + conteúdo dos dados + variáveis + espaço de busca."""
    payload = json.dumps({"split": split_hash(splits), "data": _data_hash(data),
                          "features": list(data.X.columns), "space": space}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _subset(groups: dict, idx) -> dict:
    return {a: g[idx] for a, g in groups.items()}


def tune_model(cfg: dict, seed: int, name: str, n_trials: int) -> dict:
    """Roda (ou retoma) o estudo de um modelo e avalia o melhor trial. Executável em subprocesso."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # vale também nos subprocessos
    data, splits = _prepare(cfg, seed)
    space = cfg["search_spaces"][name]
    n_jobs = cfg["optimization"]["n_jobs_model"]
    X, y = data.X, data.y
    tr, va, te = splits["train"], splits["val"], splits["test"]
    g_va, g_te = _subset(data.groups, va), _subset(data.groups, te)

    def fit(params):
        model = build_model(name, params, space, seed, n_jobs, data.categorical, data.numeric)
        t0 = time.perf_counter()
        model.fit(X.iloc[tr], y[tr])
        return model, time.perf_counter() - t0

    def objective(trial):
        params = suggest_params(trial, space)
        model, fit_seconds = fit(params)
        sv = predict_scores(model, X.iloc[va])
        thr, ba = best_threshold_balanced_accuracy(y[va], sv)
        m = evaluate(y[va], sv, g_va, thr)
        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("fit_seconds", fit_seconds)
        trial.set_user_attr("val_roc_auc", m["roc_auc"])
        for attr in data.groups:
            for k in ("eo_violation", "dp_difference"):
                trial.set_user_attr(f"val_{attr}_{k}", m[f"{attr}_{k}"])
        return ba

    ds = data.name
    # Um arquivo por estudo: processos paralelos não disputam o mesmo SQLite.
    db = resolve(cfg["optimization"]["storage_dir"]) / ds / f"{ARM}_{name}_seed{seed}.db"
    study = run_study(
        study_name=f"{ARM}_{name}_seed{seed}",
        db_path=db,
        objective=objective,
        n_trials=n_trials,
        directions=["maximize"],
        sampler=cfg["arms"][ARM]["sampler"],
        seed=seed,
        fingerprint=_fingerprint(data, splits, space),
    )

    best = study.best_trial
    model, fit_seconds = fit(best.params)
    sv, st = predict_scores(model, X.iloc[va]), predict_scores(model, X.iloc[te])
    thr, ba_val = best_threshold_balanced_accuracy(y[va], sv)
    if not np.isclose(ba_val, best.value, atol=1e-6):
        print(f"AVISO [{name}]: retreino não reproduziu o trial ({ba_val:.6f} vs {best.value:.6f}).", flush=True)

    row = {
        "dataset": ds, "arm": ARM, "model": name, "seed": seed,
        "n_trials": sum(t.state.name == "COMPLETE" for t in study.trials),
        "best_trial": best.number, "threshold": thr, "fit_seconds": fit_seconds,
    }
    row.update({f"val_{k}": v for k, v in evaluate(y[va], sv, g_va, thr).items()})
    row.update({f"test_{k}": v for k, v in evaluate(y[te], st, g_te, thr).items()})
    row["params"] = json.dumps(best.params, sort_keys=True)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.insert(0, "model", name)
    return {"row": row, "trials": trials, "scores_val": sv, "scores_test": st}


def run_a0(cfg: dict, seed: int, models: list[str] | None = None, n_trials: int | None = None) -> pd.DataFrame:
    models = models or cfg["models"]
    n_trials = n_trials or cfg["optimization"]["n_trials"]
    data, splits = _prepare(cfg, seed)
    ds = data.name
    out_dir = resolve(cfg["output"]["results_dir"])
    summary_split = split_summary(data.y, data.groups, splits)
    print(summary_split.round(4).to_string(index=False), flush=True)

    workers = max(1, min(cfg["optimization"].get("parallel_models", 1), len(models)))
    if workers == 1:
        results = [tune_model(cfg, seed, m, n_trials) for m in models]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(tune_model, cfg, seed, m, n_trials) for m in models]
            results = [f.result() for f in futures]

    summary = pd.DataFrame([r["row"] for r in results])
    trials = pd.concat([r["trials"] for r in results], ignore_index=True)

    # Escores de validação e teste: insumo para o pós-processamento (A4) e para o bootstrap.
    frames = []
    for part in ("val", "test"):
        idx = splits[part]
        f = pd.DataFrame({"row_id": idx, "split": part, "y": data.y[idx]})
        for attr, g in data.groups.items():
            f[f"group_{attr}"] = g[idx]
        for m, r in zip(models, results):
            f[f"score_{m}"] = r[f"scores_{part}"]
        frames.append(f)
    scores = pd.concat(frames, ignore_index=True)

    arm_dir = out_dir / ARM / ds
    score_dir = out_dir / "scores" / ds
    arm_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(arm_dir / f"summary_seed{seed}.csv", index=False)
    trials.to_csv(arm_dir / f"trials_seed{seed}.csv", index=False)
    scores.to_csv(score_dir / f"{ARM}_seed{seed}.csv.gz", index=False)
    summary_split.to_csv(arm_dir / f"split_seed{seed}.csv", index=False)
    print(f"Resultados em {arm_dir}", flush=True)
    return summary
