"""Baixa a base Default of Credit Card Clients (UCI id 350) para data/raw/taiwan/.

Confere o SHA-256 do .zip com o valor registrado em configs/taiwan.yaml (se houver).

    python scripts/download_taiwan.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import PurePosixPath

from faircredit.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/taiwan.yaml")
    ap.add_argument("--force", action="store_true", help="baixa de novo mesmo se o arquivo existir")
    args = ap.parse_args()

    d = load_config(args.config)["dataset"]
    raw_dir = resolve(d["raw_dir"])
    target = raw_dir / d["file"]
    if target.exists() and not args.force:
        print(f"Já existe: {target}")
        return

    print(f"Baixando {d['url']} ...")
    req = urllib.request.Request(d["url"], headers={"User-Agent": "faircredit-download/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()

    digest = hashlib.sha256(payload).hexdigest()
    print(f"SHA-256: {digest}")
    if d.get("sha256") and digest != d["sha256"]:
        raise SystemExit(f"SHA-256 diferente do registrado ({d['sha256']}). Arquivo não salvo.")
    if not d.get("sha256"):
        print("Aviso: nenhum SHA-256 registrado no YAML; registre o valor acima.")

    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        # Extrai só a planilha esperada, pelo nome-base (evita caminhos arbitrários do zip).
        members = [m for m in zf.namelist() if PurePosixPath(m).name == d["file"]]
        if not members:
            raise SystemExit(f"{d['file']} não encontrado no zip: {zf.namelist()}")
        target.write_bytes(zf.read(members[0]))
    print(f"Salvo em {target}")


if __name__ == "__main__":
    main()
