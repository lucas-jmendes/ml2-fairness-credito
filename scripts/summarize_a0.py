"""Agrega os resultados do A0 de todas as sementes: média, desvio-padrão, mínimo e máximo
de cada métrica de teste, por modelo.

    python scripts/summarize_a0.py --dataset taiwan
    python scripts/summarize_a0.py --dataset hmda

Saída: results/a0/<base>/aggregate.csv (formato longo) e aggregate_wide.csv (média ± dp).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from faircredit.config import resolve


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    arm_dir = resolve(args.results_dir) / "a0" / args.dataset
    files = sorted(arm_dir.glob("summary_seed*.csv"))
    if not files:
        raise SystemExit(f"Nenhum summary_seed*.csv em {arm_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    seeds = sorted(df["seed"].unique())

    metrics = [c for c in df.columns if c.startswith("test_") and pd.api.types.is_numeric_dtype(df[c])
               and not any(k in c for k in ("_tpr_", "_fpr_", "_selection_rate_"))]
    long = (df.groupby("model")[metrics].agg(["mean", "std", "min", "max"])
              .stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "metric"}))
    long.insert(1, "n_seeds", len(seeds))
    long.to_csv(arm_dir / "aggregate.csv", index=False)

    wide = df.groupby("model")[metrics].agg(["mean", "std"])
    fmt = pd.DataFrame({m: wide[(m, "mean")].map("{:.4f}".format) + " ± " + wide[(m, "std")].map("{:.4f}".format)
                        for m in metrics})
    fmt.to_csv(arm_dir / "aggregate_wide.csv")
    print(f"A0 | {args.dataset} | sementes {seeds}")
    print(fmt.T.to_string())


if __name__ == "__main__":
    main()
