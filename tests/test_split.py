"""Divisão 60/20/20: disjunta, completa, determinística e estratificada por grupo x rótulo."""

import numpy as np

from faircredit.split import stratified_split

rng = np.random.default_rng(1)
N = 30000
S = rng.choice(["female", "male"], size=N, p=[0.6, 0.4])
Y = rng.binomial(1, np.where(S == "female", 0.21, 0.24))
STRATA = np.char.add(S.astype(str), Y.astype(str))


def test_partition_is_complete_and_disjoint():
    sp = stratified_split(STRATA, seed=0)
    allidx = np.concatenate(list(sp.values()))
    assert len(allidx) == N and len(np.unique(allidx)) == N


def test_proportions():
    sp = stratified_split(STRATA, seed=0)
    for name, frac in [("train", 0.6), ("val", 0.2), ("test", 0.2)]:
        assert abs(len(sp[name]) / N - frac) < 0.001


def test_strata_preserved():
    sp = stratified_split(STRATA, seed=0)
    strata = STRATA
    full = {k: np.mean(strata == k) for k in np.unique(strata)}
    for idx in sp.values():
        for k, p in full.items():
            assert abs(np.mean(strata[idx] == k) - p) < 0.002


def test_deterministic_and_seed_dependent():
    a, b, c = stratified_split(STRATA, 3), stratified_split(STRATA, 3), stratified_split(STRATA, 4)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test"], c["test"])
