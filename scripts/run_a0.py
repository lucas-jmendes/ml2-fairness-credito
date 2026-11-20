"""Executa o braço A0 (modelos sem mitigação).

Exemplos:
    python scripts/run_a0.py --seeds 0
    python scripts/run_a0.py --seeds 0 1 2 3 4
    python scripts/run_a0.py --seeds 0 --models lightgbm --n-trials 5   # teste rápido
    python scripts/run_a0.py --config configs/hmda.yaml --seeds 0
"""

from __future__ import annotations

import argparse
import os
import sys

import optuna

from faircredit.arms.a0 import run_a0
from faircredit.config import load_config

COLS = ["model", "threshold", "val_balanced_accuracy", "test_balanced_accuracy", "test_roc_auc"]
FAIR = ["eo_violation", "tpr_diff", "fpr_diff", "dp_difference"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/taiwan.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="padrão: todas as do YAML")
    ap.add_argument("--models", nargs="+", default=None, help="padrão: os cinco do YAML")
    ap.add_argument("--n-trials", type=int, default=None, help="padrão: optimization.n_trials do YAML")
    args = ap.parse_args()

    # Acentos corretos no Windows, inclusive nos subprocessos (que herdam a variável).
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config(args.config)
    for seed in args.seeds or cfg["seeds"]:
        print(f"\n=== A0 | {cfg['dataset']['name']} | semente {seed} ===", flush=True)
        summary = run_a0(cfg, seed, args.models, args.n_trials)
        print(summary[COLS].round(4).to_string(index=False), flush=True)
        for attr in cfg["dataset"]["sensitive"]:
            print(f"\nJustiça no teste — atributo: {attr}")
            print(summary[["model"] + [f"test_{attr}_{k}" for k in FAIR]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
