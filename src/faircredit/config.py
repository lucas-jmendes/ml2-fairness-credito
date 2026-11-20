"""Leitura das configurações YAML e resolução de caminhos a partir da raiz do repositório."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Caminho absoluto; caminhos relativos partem da raiz do repositório."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _merge(base: dict, override: dict) -> dict:
    """Mescla recursiva: chaves do override substituem as da base (dicts são mesclados)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: str | Path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Variantes (ex.: análises de sensibilidade) herdam de outro YAML e só mudam o necessário.
    base = cfg.pop("base_config", None)
    return _merge(_read(base), cfg) if base else cfg


def load_config(path: str | Path) -> dict:
    """Carrega a configuração da base e anexa os espaços de busca em cfg['search_spaces']."""
    cfg = _read(path)
    spaces_file = cfg.get("search_spaces_file")
    if spaces_file:
        with open(resolve(spaces_file), encoding="utf-8") as f:
            cfg["search_spaces"] = yaml.safe_load(f)
    return cfg
