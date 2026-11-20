"""Confere as métricas próprias contra scikit-learn e Fairlearn."""

import numpy as np
from fairlearn.metrics import demographic_parity_difference as fl_dp
from fairlearn.metrics import equalized_odds_difference as fl_eo
from sklearn.metrics import balanced_accuracy_score

from faircredit.metrics import (
    balanced_accuracy,
    best_threshold_balanced_accuracy,
    demographic_parity_difference,
    equalized_odds_violation,
    evaluate,
)

rng = np.random.default_rng(0)
N = 5000
GROUPS = rng.choice(["female", "male"], size=N, p=[0.6, 0.4])
Y = rng.binomial(1, np.where(GROUPS == "female", 0.2, 0.25))
SCORES = np.clip(0.3 * Y + rng.normal(0.3, 0.2, N) + 0.05 * (GROUPS == "male"), 0, 1)
PRED = (SCORES >= 0.4).astype(int)


def test_balanced_accuracy_matches_sklearn():
    assert np.isclose(balanced_accuracy(Y, PRED), balanced_accuracy_score(Y, PRED))


def test_equalized_odds_matches_fairlearn():
    assert np.isclose(
        equalized_odds_violation(Y, PRED, GROUPS),
        fl_eo(Y, PRED, sensitive_features=GROUPS, agg="worst_case"),
    )


def test_demographic_parity_matches_fairlearn():
    assert np.isclose(demographic_parity_difference(PRED, GROUPS), fl_dp(Y, PRED, sensitive_features=GROUPS))


def test_best_threshold_is_optimal_on_grid():
    thr, ba = best_threshold_balanced_accuracy(Y, SCORES)
    assert np.isclose(ba, balanced_accuracy_score(Y, (SCORES >= thr).astype(int)))
    grid = np.unique(SCORES)
    best_grid = max(balanced_accuracy_score(Y, (SCORES >= t).astype(int)) for t in grid[::25])
    assert ba >= best_grid - 1e-12


def test_evaluate_consistent():
    m = evaluate(Y, SCORES, {"sex": GROUPS}, 0.4)
    assert np.isclose(m["sex_eo_violation"], max(m["sex_tpr_diff"], m["sex_fpr_diff"]))
    assert np.isclose(m["sex_tpr_diff"], abs(m["sex_tpr_female"] - m["sex_tpr_male"]))


def test_groups_with_none_are_ignored():
    g = GROUPS.astype(object).copy()
    g[:100] = None
    m = evaluate(Y, SCORES, {"sex": g}, 0.4)
    keep = np.arange(N) >= 100
    expected = fl_eo(Y[keep], PRED[keep], sensitive_features=GROUPS[keep], agg="worst_case")
    assert np.isclose(m["sex_eo_violation"], expected)


def test_multigroup_eo_is_max_minus_min():
    g = rng.choice(["a", "b", "c"], size=N)
    assert np.isclose(equalized_odds_violation(Y, PRED, g), fl_eo(Y, PRED, sensitive_features=g, agg="worst_case"))
