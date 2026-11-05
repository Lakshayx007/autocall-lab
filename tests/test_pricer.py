"""Payoff rules on hand-built paths, the Monte Carlo engine against Black-Scholes, and sanity of the note's price."""
import numpy as np
import pytest

from autocall import blackscholes as bs
from autocall import mc
from autocall.note import TermSheet, cashflows

TS = TermSheet()                       # 12 quarterly dates, 8% a year = 2 per period
CPN = TS.coupon_per_period * TS.notional


def worst_path(*levels):
    """One path of worst-of levels, padded with 0.85 (no call, coupon paid) to 12 dates."""
    w = list(levels) + [0.85] * (TS.n_obs - len(levels))
    return np.array([w])


def test_autocall_on_first_date():
    cf = cashflows(worst_path(1.02), TS)
    assert cf.call_index[0] == 0
    assert cf.redemption[0, 0] == 100 and cf.coupons[0, 0] == CPN
    assert cf.coupons[0, 1:].sum() == 0 and cf.redemption[0, 1:].sum() == 0


def test_memory_coupon_pays_missed_periods():
    w = worst_path(0.65, 0.60, 0.80)
    with_memory = cashflows(w, TS).coupons[0]
    no_memory = cashflows(w, TermSheet(memory=False)).coupons[0]
    assert with_memory[:3].tolist() == [0, 0, 3 * CPN]
    assert no_memory[:3].tolist() == [0, 0, CPN]


def test_capital_loss_below_barrier_at_maturity():
    w = worst_path(*([0.85] * 11 + [0.50]))
    cf = cashflows(w, TS)
    assert cf.knocked_in[0] and cf.redemption[0, -1] == pytest.approx(50.0) and cf.coupons[0, -1] == 0


def test_no_loss_between_barriers_at_maturity():
    cf = cashflows(worst_path(*([0.85] * 11 + [0.65])), TS)
    assert not cf.knocked_in[0] and cf.redemption[0, -1] == 100 and cf.coupons[0, -1] == 0


def test_engine_matches_black_scholes():
    m = mc.Market(vol=np.array([0.2]), corr=np.eye(1), rate=0.04, div=np.array([0.01]))
    for K in (0.8, 1.0, 1.25):
        p, se = mc.european_mc(lambda s: np.maximum(s[:, 0] - K, 0), m, 2.0, n_paths=200_000)
        assert abs(p - bs.call(1, K, 2.0, 0.04, 0.01, 0.2)) < 4 * se
        p, se = mc.european_mc(lambda s: (s[:, 0] > K).astype(float), m, 2.0, n_paths=200_000)
        assert abs(p - bs.digital_call(1, K, 2.0, 0.04, 0.01, 0.2)) < 4 * se


def test_perfect_correlation_reduces_worst_of_to_one_asset():
    one = mc.Market(vol=np.array([0.2]), corr=np.eye(1), rate=0.04, div=np.array([0.02]))
    three = mc.Market(vol=np.full(3, 0.2), corr=np.full((3, 3), 1.0) - 1e-9 * (1 - np.eye(3)), rate=0.04,
                      div=np.full(3, 0.02))
    z1 = mc.normals(50_000, TS.n_obs, 1)
    z3 = np.repeat(z1, 3, axis=2)                       # identical shocks for all three assets
    assert mc.price(TS, one, z=z1).price == pytest.approx(mc.price(TS, three, z=z3).price, abs=1e-3)


def test_price_is_reproducible_and_sensible():
    m = mc.Market(vol=np.array([0.17, 0.18, 0.24]), corr=np.array([[1, .7, .6], [.7, 1, .6], [.6, .6, 1]]),
                  rate=0.04, div=np.array([0.01, 0.028, 0.012]))
    a, b = mc.price(TS, m, n_paths=40_000), mc.price(TS, m, n_paths=40_000)
    assert a.price == b.price
    assert 80 < a.price < 110 and a.ci95[0] < a.price < a.ci95[1]
    assert abs(a.call_prob.sum() + a.maturity_prob - 1) < 1e-12
    z = mc.normals(40_000, TS.n_obs, 3)
    assert mc.price(TermSheet(coupon_pa=0.10), m, z=z).price > mc.price(TS, m, z=z).price
    assert mc.price(TermSheet(ki_barrier=0.50), m, z=z).price > mc.price(TS, m, z=z).price
