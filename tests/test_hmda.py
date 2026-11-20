"""Conversão da razão dívida/renda publicada pela HMDA."""

import numpy as np
import pandas as pd

from faircredit.hmda import parse_dti


def test_parse_dti():
    s = pd.Series(["<20%", "20%-<30%", "30%-<36%", "42", "50%-60%", ">60%", "Exempt", None], dtype="string")
    out = parse_dti(s).to_numpy(dtype=float)
    np.testing.assert_allclose(out[:6], [15, 25, 33, 42, 55, 65])
    assert np.isnan(out[6]) and np.isnan(out[7])
