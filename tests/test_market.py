"""Market data: calendars, returns, volatility and correlation estimators, and the cached snapshot."""
import numpy as np
import pandas as pd
import pytest

from autocall import blackscholes as bs
from autocall import market as M


@pytest.fixture(scope="module")
def rets():
    return M.log_returns(M.common_closes(M.load_prices()))


def test_common_calendar_has_no_gaps(rets):
    closes = M.common_closes(M.load_prices())
    assert closes.notna().all().all()
    assert len(rets) == len(closes) - 1
    assert np.isfinite(rets.values).all()


def test_historical_vol_recovers_a_known_vol():
    rng = np.random.default_rng(1)
    daily = 0.2 / np.sqrt(M.TRADING_DAYS)
    r = pd.DataFrame(rng.normal(0, daily, (20_000, 2)), columns=["a", "b"])
    assert M.historical_vol(r).sub(0.2).abs().max() < 0.005


def test_ewma_follows_its_recursion():
    r = pd.DataFrame({"a": [0.01, -0.02, 0.03, 0.0, 0.01] * 10})
    v = M.ewma_vol(r, lam=0.9)["a"].values ** 2 / M.TRADING_DAYS
    for t in range(1, len(r)):
        assert v[t] == pytest.approx(0.9 * v[t - 1] + 0.1 * r["a"].iloc[t - 1] ** 2)


def test_correlation_horizon_sums_non_overlapping_blocks():
    rng = np.random.default_rng(2)
    r = pd.DataFrame(rng.normal(size=(1_000, 2)), columns=["a", "b"])
    r["b"] = r["a"].shift(1).fillna(0)               # b reacts one day late to a
    assert abs(M.correlation(r, 1).iloc[0, 1]) < 0.1  # same-day returns miss the link
    assert M.correlation(r, 5).iloc[0, 1] > 0.7       # weekly returns recover most of it


def test_snapshot_is_usable(rets):
    s = M.snapshot()
    assert s.names == list(M.INDICES)
    assert np.allclose(s.corr, s.corr.T) and np.allclose(np.diag(s.corr), 1)
    ok, eig = M.is_positive_definite(s.corr)
    assert ok and eig.min() > 0
    np.linalg.cholesky(s.corr)
    assert (s.vol > 0.05).all() and (s.vol < 0.6).all()
    assert 0 < s.rate < 0.1 and (s.div >= 0).all()


def test_closing_time_mismatch_shows_in_the_data(rets):
    daily = M.correlation(rets, 1).loc["S&P 500", "Nikkei 225"]
    weekly = M.correlation(rets, 5).loc["S&P 500", "Nikkei 225"]
    assert weekly > daily + 0.2


def test_black_scholes_parity_and_implied_vol_round_trip():
    S, K, T, r, q, v = 100, 95, 2.0, 0.04, 0.015, 0.22
    c, p = bs.call(S, K, T, r, q, v), bs.put(S, K, T, r, q, v)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T))
    assert bs.implied_vol(c, S, K, T, r, q, True) == pytest.approx(v, abs=1e-8)


def test_spx_implied_quotes_agree():
    imp = M.load_meta().get("spx_implied")
    if imp is None:
        pytest.skip("option quotes were not available at fetch time")
    ivs = [q["iv"] for q in imp["quotes"]]
    assert max(ivs) - min(ivs) < 0.03 and 0.05 < imp["atm_iv"] < 0.6
