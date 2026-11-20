"""Preparo da base HMDA (um estado, um ano): filtros, rótulo, variáveis e subamostra.

Rótulo: 1 = pedido negado (action_taken 3); 0 = aprovado (1 = contratado, 2 = aprovado e não
aceito pelo solicitante). Pedidos retirados, incompletos ou comprados não entram.

Variáveis de entrada (proposta, seção 5): valor do empréstimo, renda, razão dívida/renda,
razão empréstimo/valor do imóvel, tipo e finalidade do empréstimo e prazo. Campos que só
existem depois da decisão (taxa de juros, custos, motivo da negação) ficam de fora, porque
vazariam o rótulo. Variáveis geográficas também ficam de fora (forte proxy de raça).
"""

from __future__ import annotations

import pandas as pd

# Faixas publicadas de dívida/renda -> valor numérico ordinal (ponto médio aproximado).
# Valores entre 36 e 49 já vêm como número. "Exempt"/ausente -> NaN.
DTI_BINS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}


def parse_dti(s: pd.Series) -> pd.Series:
    s = s.astype(object)
    binned = s.map(DTI_BINS).astype(float)
    numeric = pd.to_numeric(s, errors="coerce").astype(float)
    return binned.fillna(numeric)


def prepare_hmda(raw: pd.DataFrame, d: dict) -> tuple[pd.DataFrame, dict]:
    """Devolve a base preparada e um relatório com a contagem de linhas em cada etapa."""
    report = {"bruto": len(raw)}
    df = raw
    for col, allowed in d["filters"].items():
        df = df[df[col].isin(allowed)]
        report[f"filtro {col} in {allowed}"] = len(df)

    t = d["target"]
    df = df[df[t["column"]].isin(t["positive"] + t["negative"])].copy()
    df["y"] = df[t["column"]].isin(t["positive"]).astype(int)

    # Linhas sem grupo definido no atributo de estratificação não podem ser usadas.
    for attr in d["stratify_by"]:
        spec = d["sensitive"][attr]
        df = df[df[spec["column"]].isin(list(spec["map"]))]
        report[f"{attr} definido"] = len(df)

    df["debt_to_income_ratio"] = parse_dti(df["debt_to_income_ratio"].astype("string"))
    for col in ("loan_amount", "income", "loan_to_value_ratio", "loan_term"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = list(d["features"]) + ["y"] + [s["column"] for s in d["sensitive"].values()]
    df = df[keep]

    # Opcional (análise de sensibilidade): descarta linhas com ausentes nestas variáveis.
    for col in d.get("drop_missing") or []:
        df = df[df[col].notna()]
        report[f"sem ausentes em {col}"] = len(df)

    n = d.get("sample_size")
    if n and n < len(df):
        # Subamostra estratificada por rótulo x atributos de estratificação, semente fixa:
        # a base preparada é a mesma em todas as repetições do experimento.
        key = df["y"].astype(str)
        for attr in d["stratify_by"]:
            key = key + "|" + df[d["sensitive"][attr]["column"]].astype(str)
        frac = n / len(df)
        df = (df.groupby(key, group_keys=False)
                .sample(frac=frac, random_state=d["sample_seed"]))
        report["subamostra"] = len(df)
    df = df.reset_index(drop=True)
    report["taxa de negação"] = float(df["y"].mean())
    return df, report


def missing_report(df: pd.DataFrame, features: list[str]) -> pd.Series:
    return df[features].isna().mean().round(4)

