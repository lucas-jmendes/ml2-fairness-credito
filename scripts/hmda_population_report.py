"""Relatório da população HMDA excluída pelo recorte racial (insumo da seção de limitações).

1. Consulta a API de agregações do CFPB: pedidos com decisão (ação 1, 2 ou 3) por raça
   derivada no estado/ano da configuração, com taxa de negação.
2. Baixa os pedidos com raça derivada "Joint" e cruza a raça do solicitante com a do
   co-solicitante. A documentação pública não descreve a regra de derivação; ela é verificada
   empiricamente.
3. Cruza o sexo derivado com o sexo do solicitante e do co-solicitante (arquivo principal).

    python scripts/hmda_population_report.py

Saídas em results/data/: hmda_racas.csv, hmda_joint_raca.csv, hmda_joint_sexo.csv.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

import pandas as pd

from faircredit.config import load_config, resolve

API = "https://ffiec.cfpb.gov/v2/data-browser-api/view"
RACES = ["White", "Black or African American", "Race Not Available", "Asian", "Joint",
         "American Indian or Alaska Native", "2 or more minority races",
         "Native Hawaiian or Other Pacific Islander", "Free Form Text Only"]
# Códigos da LAR (raça do solicitante, campo 1), agrupados nas categorias principais.
RACE_CODES = {1: "Indígena (AmInd/AK)", 2: "Asiática", 3: "Negra", 4: "Havaiana/Ilhas do Pacífico",
              5: "Branca", 6: "Não informada", 7: "Não se aplica", 8: "Sem co-solicitante"}
RACE_CODES.update({c: "Asiática" for c in range(21, 28)})
RACE_CODES.update({c: "Havaiana/Ilhas do Pacífico" for c in range(41, 45)})
SEX_CODES = {1: "Masculino", 2: "Feminino", 3: "Não informado", 4: "Não se aplica",
             5: "Sem co-solicitante", 6: "Ambos marcados"}


def _get(url: str) -> bytes:
    # Sem User-Agent personalizado: a API do CFPB recusa (403) identificadores próprios.
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config("configs/hmda.yaml")
    d = cfg["dataset"]
    q = urllib.parse.parse_qs(urllib.parse.urlparse(d["url"]).query)
    state, year = q["states"][0], q["years"][0]
    out = resolve(cfg["output"]["results_dir"]) / "data"
    out.mkdir(parents=True, exist_ok=True)

    params = {"states": state, "years": year, "actions_taken": "1,2,3", "races": ",".join(RACES)}
    agg = json.loads(_get(f"{API}/aggregations?{urllib.parse.urlencode(params)}"))["aggregations"]
    df = pd.DataFrame(agg)
    df["denied"] = (df["actions_taken"] == "3") * df["count"]
    races = df.groupby("races")[["count", "denied"]].sum().sort_values("count", ascending=False)
    races["share"] = races["count"] / races["count"].sum()
    races["denial_rate"] = races["denied"] / races["count"]
    races["in_study"] = races.index.isin(list(d["sensitive"]["race"]["map"]))
    races.to_csv(out / "hmda_racas.csv")
    print(f"Pedidos com decisão por raça derivada — {state} {year}\n{races.round(4).to_string()}\n")

    joint_path = resolve(d["raw_dir"]) / f"hmda_{state}_{year}_joint.csv"
    if not joint_path.exists():
        params = {"states": state, "years": year, "actions_taken": "1,2,3", "races": "Joint"}
        joint_path.write_bytes(_get(f"{API}/csv?{urllib.parse.urlencode(params)}"))
    j = pd.read_csv(joint_path, low_memory=False)
    ct = pd.crosstab(j["applicant_race-1"].map(RACE_CODES).rename("solicitante"),
                     j["co-applicant_race-1"].map(RACE_CODES).rename("co-solicitante"))
    ct.to_csv(out / "hmda_joint_raca.csv")
    white = "Branca"
    mixed = (ct.loc[white].drop(white, errors="ignore").sum() + ct[white].drop(white, errors="ignore").sum())
    print(f"Raça derivada 'Joint' (n={len(j)}): solicitante x co-solicitante (raça 1)\n{ct}\n"
          f"Um branco + outro de minoria (pelo 1º campo de raça): {mixed / ct.values.sum():.1%}\n")

    raw = pd.read_csv(resolve(d["raw_dir"]) / d["raw_file"],
                      usecols=["derived_sex", "applicant_sex", "co-applicant_sex"])
    cs = pd.crosstab([raw["applicant_sex"].map(SEX_CODES).rename("solicitante"),
                      raw["co-applicant_sex"].map(SEX_CODES).rename("co-solicitante")], raw["derived_sex"])
    cs.to_csv(out / "hmda_joint_sexo.csv")
    print(f"Sexo derivado x sexo do solicitante/co-solicitante\n{cs}")


if __name__ == "__main__":
    main()
