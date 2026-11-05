"""Monte Carlo engine: correlated geometric Brownian motion sampled on the observation dates, and the note's price."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .note import TermSheet, cashflows

DEFAULT_PATHS = 100_000
DEFAULT_SEED = 7


@dataclass
class Market:
    """Pricing inputs for n assets (spot levels only matter through performance, so they are not needed)."""
    vol: np.ndarray
    corr: np.ndarray
    rate: float
    div: np.ndarray

    @classmethod
    def from_snapshot(cls, snap) -> "Market":
        return cls(vol=np.asarray(snap.vol, float), corr=np.asarray(snap.corr, float), rate=float(snap.rate),
                   div=np.asarray(snap.div, float))


def normals(n_paths: int, n_steps: int, n_assets: int, seed: int = DEFAULT_SEED, antithetic: bool = True) -> np.ndarray:
    """Independent standard normals (paths x steps x assets). With antithetic variates the second half mirrors the
    first, which cancels much of the noise for payoffs that move steadily with the market."""
    rng = np.random.default_rng(seed)
    if not antithetic:
        return rng.standard_normal((n_paths, n_steps, n_assets))
    half = rng.standard_normal(((n_paths + 1) // 2, n_steps, n_assets))
    return np.concatenate([half, -half])[:n_paths]


def performance_paths(mkt: Market, times: np.ndarray, z: np.ndarray) -> np.ndarray:
    """S_i(t) / S_i(0) on each time, under the risk-neutral measure. Exact GBM steps, so only the observation dates
    are simulated: log S(t + dt) = log S(t) + (r - q - vol^2 / 2) dt + vol sqrt(dt) * (L @ z), with L L' = corr."""
    L = np.linalg.cholesky(mkt.corr)
    dt = np.diff(np.concatenate([[0.0], times]))
    drift = (mkt.rate - mkt.div - 0.5 * mkt.vol ** 2)[None, None, :] * dt[None, :, None]
    shocks = (z @ L.T) * mkt.vol[None, None, :] * np.sqrt(dt)[None, :, None]
    return np.exp(np.cumsum(drift + shocks, axis=1))


@dataclass
class PriceResult:
    price: float                  # % of notional (notional 100)
    stderr: float
    ci95: tuple[float, float]
    n_paths: int
    call_prob: np.ndarray         # probability of autocall on each observation date (last entry: 0, maturity)
    maturity_prob: float          # probability the note runs to maturity
    loss_prob: float              # probability of a capital loss at maturity
    expected_life: float          # years
    pv_coupons: float
    pv_redemption: float
    expected_coupons: float       # number of coupon periods paid, on average
    loss_given_loss: float        # average redemption when there is a loss (% of notional)


def _pair_stderr(x: np.ndarray, antithetic: bool) -> float:
    """Standard error of the mean; with antithetic variates the pair averages are the independent samples."""
    if antithetic and len(x) % 2 == 0:
        h = len(x) // 2
        x = 0.5 * (x[:h] + x[h:])
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def price(ts: TermSheet, mkt: Market, n_paths: int = DEFAULT_PATHS, seed: int = DEFAULT_SEED,
          antithetic: bool = True, z: np.ndarray | None = None, spot_rel: np.ndarray | None = None) -> PriceResult:
    """Present value of the note per 100 notional, with Monte Carlo diagnostics.

    z: reuse random numbers (common random numbers make bump-and-reprice sensitivities stable).
    spot_rel: today's level of each index relative to its initial fixing (default 1.0, i.e. at issue).
    """
    times = ts.times
    if z is None:
        z = normals(n_paths, len(times), len(mkt.vol), seed, antithetic)
    perf = performance_paths(mkt, times, z)
    if spot_rel is not None:
        perf = perf * np.asarray(spot_rel, float)[None, None, :]
    worst = perf.min(axis=2)
    cf = cashflows(worst, ts)
    df = np.exp(-mkt.rate * times)
    pv_cpn = cf.coupons @ df
    pv_red = cf.redemption @ df
    pv = pv_cpn + pv_red
    se = _pair_stderr(pv, antithetic)
    n = len(pv)
    called = cf.call_index >= 0
    call_prob = np.array([(cf.call_index == k).mean() for k in range(len(times))])
    life = np.where(called, times[np.clip(cf.call_index, 0, None)], times[-1])
    lost = cf.knocked_in
    return PriceResult(
        price=float(pv.mean()), stderr=se, ci95=(float(pv.mean() - 1.96 * se), float(pv.mean() + 1.96 * se)),
        n_paths=n, call_prob=call_prob, maturity_prob=float((~called).mean()), loss_prob=float(lost.mean()),
        expected_life=float(life.mean()), pv_coupons=float(pv_cpn.mean()), pv_redemption=float(pv_red.mean()),
        expected_coupons=float(cf.coupon_units.sum(axis=1).mean()),
        loss_given_loss=float(cf.redemption[lost, -1].mean()) if lost.any() else float("nan"),
    )


# ── validation: the same path engine on payoffs with closed-form prices ─────
def european_mc(payoff, mkt: Market, T: float, n_paths: int = DEFAULT_PATHS, seed: int = DEFAULT_SEED,
                antithetic: bool = True) -> tuple[float, float]:
    """Discounted mean of payoff(performance at T) with its standard error, using the note's path engine."""
    z = normals(n_paths, 1, len(mkt.vol), seed, antithetic)
    perf = performance_paths(mkt, np.array([T]), z)[:, -1, :]
    pv = np.exp(-mkt.rate * T) * payoff(perf)
    return float(pv.mean()), _pair_stderr(pv, antithetic)
