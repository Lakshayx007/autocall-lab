"""Sensitivities and the fair-coupon solve."""
from dataclasses import replace

import numpy as np
import pytest

from autocall import mc
from autocall import risk as R
from autocall.note import TermSheet

TS = TermSheet()
MKT = mc.Market(vol=np.array([0.15, 0.20, 0.30]), corr=np.array([[1, .7, .6], [.7, 1, .6], [.6, .6, 1]]),
                rate=0.04, div=np.array([0.01, 0.028, 0.012]))
NAMES = ["low vol", "mid vol", "high vol"]
N = 40_000


@pytest.fixture(scope="module")
def g():
    return R.greeks(TS, MKT, NAMES, n_paths=N)


def test_note_gains_when_indices_rise_most_from_the_likely_laggard(g):
    d = g.loc[NAMES, "delta per 1% move"]
    assert (d > 0).all()
    assert d.idxmax() == "high vol"                    # the most volatile index is most often the worst


def test_note_loses_value_when_volatility_rises(g):
    assert g.loc["All three together", "vega per vol point"] < 0


def test_higher_correlation_helps_the_investor():
    s = R.correlation_sensitivity(TS, MKT, n_paths=N)
    assert s.is_monotonic_increasing


def test_fair_coupon_hits_the_target_price():
    z = mc.normals(N, TS.n_obs, 3)
    c = R.fair_coupon(TS, MKT, margin=2.0, z=z)
    assert mc.price(replace(TS, coupon_pa=c), MKT, z=z).price == pytest.approx(TS.issue_price - 2.0, abs=1e-3)
    assert R.fair_coupon(TS, MKT, margin=0.0, z=z) > c   # less margin, more coupon for the investor


def test_riskier_barrier_pays_a_higher_coupon():
    grid = R.coupon_grid(TS, MKT, 2.0, "ki_barrier", [0.5, 0.6, 0.7], n_paths=N)
    assert grid.is_monotonic_increasing


def test_correlation_shift_stays_valid():
    c = R.shift_correlation(MKT.corr, 0.2)
    assert np.allclose(np.diag(c), 1) and np.linalg.eigvalsh(c).min() > 0
    with pytest.raises(np.linalg.LinAlgError):
        R.shift_correlation(MKT.corr, -1.6)        # every pair near -1: impossible for 3 assets
