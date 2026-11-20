"""Estudos do Optuna em SQLite, retomáveis após interrupção."""

from __future__ import annotations

from pathlib import Path

import optuna
from optuna.trial import TrialState

SAMPLERS = {
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed),
    "nsga2": lambda seed: optuna.samplers.NSGAIISampler(seed=seed),
    "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
}


def storage_url(db_path: Path) -> str:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.resolve().as_posix()}"


def run_study(
    study_name: str,
    db_path: Path,
    objective,
    n_trials: int,
    directions: list[str],
    sampler: str,
    seed: int,
    fingerprint: str,
    log_every: int = 10,
) -> optuna.Study:
    """Cria ou retoma o estudo e executa só os trials que faltam para chegar a n_trials.

    Trials interrompidos (estado RUNNING/FAIL) não contam; a retomada completa o orçamento.
    Observação: a sequência de sugestões após uma retomada não é idêntica à de uma
    execução contínua, mas o orçamento e o espaço de busca são os mesmos.

    'fingerprint' resume divisão, variáveis e espaço de busca. Um estudo salvo com outra
    impressão digital não é retomado: é preciso apagar o .db (evita misturar trials de
    configurações diferentes).
    """
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url(db_path),
        directions=directions,
        sampler=SAMPLERS[sampler](seed),
        load_if_exists=True,
    )
    saved = study.user_attrs.get("fingerprint")
    if saved is None and not study.trials:
        study.set_user_attr("fingerprint", fingerprint)
    elif saved != fingerprint:
        raise RuntimeError(
            f"[{study_name}] o estudo em {db_path} foi criado com outra configuração "
            f"(divisão, variáveis ou espaço de busca). Apague o arquivo para recomeçar."
        )
    done = sum(t.state == TrialState.COMPLETE for t in study.trials)
    remaining = n_trials - done
    if remaining <= 0:
        print(f"[{study_name}] já concluído ({done} trials).", flush=True)
        return study

    def _log(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
        n = sum(t.state == TrialState.COMPLETE for t in st.trials)
        if n % log_every == 0 or n == n_trials:
            best = st.best_value if len(directions) == 1 else len(st.best_trials)
            print(f"[{study_name}] {n}/{n_trials} trials | melhor: {best}", flush=True)

    print(f"[{study_name}] {done} concluídos, executando {remaining}.", flush=True)
    study.optimize(objective, n_trials=remaining, callbacks=[_log])
    return study
