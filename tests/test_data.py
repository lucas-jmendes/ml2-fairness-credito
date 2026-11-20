"""Garantias do carregamento: atributo sensível nunca nas entradas; herança de configuração."""

import numpy as np
import pandas as pd
import pytest

from faircredit import data as data_mod
from faircredit.config import load_config


def _fake_cfg(features):
    return {"dataset": {
        "name": "fake", "loader": "fake",
        "sensitive": {"sex": {"column": "SEX", "map": {1: "male", 2: "female"}},
                      "age": {"column": "AGE", "below": 25, "labels": ["lt25", "ge25"]}},
        "stratify_by": ["sex", "age"], "exclude_from_features": ["SEX", "AGE"],
        "categorical": [], "features": features,
    }}


@pytest.fixture
def fake_reader(monkeypatch):
    df = pd.DataFrame({"SEX": [1, 2, 1, 2], "AGE": [22, 40, 30, 24], "x": [0.1, 0.2, 0.3, 0.4]})
    monkeypatch.setitem(data_mod.READERS, "fake", lambda d: (df, np.array([0, 1, 0, 1]), list(d["features"])))


def test_sensitive_in_features_raises(fake_reader):
    with pytest.raises(AssertionError):
        data_mod.load_dataset(_fake_cfg(["x", "SEX"]))
    with pytest.raises(AssertionError):
        data_mod.load_dataset(_fake_cfg(["x", "AGE"]))


def test_groups_and_strata(fake_reader):
    ds = data_mod.load_dataset(_fake_cfg(["x"]))
    assert list(ds.X.columns) == ["x"]
    assert list(ds.groups["sex"]) == ["male", "female", "male", "female"]
    assert list(ds.groups["age"]) == ["lt25", "ge25", "ge25", "lt25"]
    assert ds.strata[0] == "0|male|lt25"


def test_real_configs_never_use_sensitive_attributes():
    for path in ["configs/taiwan.yaml", "configs/hmda.yaml", "configs/hmda_sens_ltv.yaml"]:
        d = load_config(path)["dataset"]
        sensitive_cols = {s["column"] for s in d["sensitive"].values()}
        assert not sensitive_cols & set(d.get("features", [])), path
        if d["loader"] == "taiwan":
            assert sensitive_cols <= set(d["exclude_from_features"])


def test_config_inheritance():
    base, variant = load_config("configs/hmda.yaml"), load_config("configs/hmda_sens_ltv.yaml")
    assert variant["dataset"]["name"] == "hmda_sens_ltv"
    assert variant["dataset"]["drop_missing"] == ["loan_to_value_ratio"]
    assert variant["dataset"]["features"] == base["dataset"]["features"]
    assert variant["search_spaces"] == base["search_spaces"]
