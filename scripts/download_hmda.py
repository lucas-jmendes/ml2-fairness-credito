"""Baixa os dados públicos da HMDA (API do CFPB) e gera a base preparada.

    python scripts/download_hmda.py                  # baixa (se preciso) e prepara
    python scripts/download_hmda.py --force          # baixa de novo
    python scripts/download_hmda.py --config configs/hmda_sens_ltv.yaml   # variante de sensibilidade

Saídas: data/raw/hmda/<raw_file> (bruto) e data/processed/<arquivo> (preparado), além de
results/data/hmda_preparo.csv com as contagens de cada etapa de filtragem.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request

import pandas as pd

from faircredit.config import load_config, resolve
from faircredit.hmda import missing_report, prepare_hmda


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hmda.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["dataset"]

    raw = resolve(d["raw_dir"]) / d["raw_file"]
    if args.force or not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando {d['url']} ...")
        # Sem User-Agent personalizado: a API do CFPB recusa (403) identificadores próprios.
        req = urllib.request.Request(d["url"])
        with urllib.request.urlopen(req, timeout=1200) as resp, open(raw, "wb") as f:
            shutil.copyfileobj(resp, f)

    digest = sha256(raw)
    print(f"SHA-256: {digest}")
    if d.get("sha256") and digest != d["sha256"]:
        print("AVISO: SHA-256 diferente do registrado no YAML. O CFPB pode ter republicado os "
              "dados; os resultados podem não ser idênticos aos do artigo.")

    df, report = prepare_hmda(pd.read_csv(raw, low_memory=False), d)
    out = resolve(d["processed_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    rep = pd.Series(report, name="linhas")
    rep_dir = resolve(cfg["output"]["results_dir"]) / "data"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep.to_csv(rep_dir / f"{d['name']}_preparo.csv")
    print(rep.to_string())
    print("\nProporção de ausentes por variável:\n" + missing_report(df, d["features"]).to_string())
    print(f"\nSalvo em {out} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
