"""The issuer's side: sell the note, delta-hedge it with the three indices, and measure what is left.

At each rebalancing date the note is revalued by Monte Carlo from the current state (index levels, coupons owed under
the memory feature), deltas are taken by bump and reprice, and the hedge is adjusted. Index holdings earn their
dividend yield; cash earns the risk-free rate. Scenarios are simulated under the same model used for pricing, so the
average profit equals the margin and the spread of profits shows how well delta hedging works.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import mc
from .note import TermSheet, cashflows


def remaining_value(ts: TermSheet, mkt: mc.Market, t: float, perf_now: np.ndarray, missed: float,
                    z: np.ndarray) -> float:
    """Value at time t (between observation dates) of the note's remaining cash flows, given today's performance of
    each index (level / initial fixing) and the coupons owed. z: inner normals (paths x n_obs x assets)."""
    rem = ts.times[ts.times > t + 1e-9]
    start = ts.n_obs - len(rem)
    rel = mc.performance_paths(mkt, rem - t, z[:, :len(rem), :])
    worst = (rel * perf_now[None, None, :]).min(axis=2)
    cf = cashflows(worst, ts, start=start, missed0=missed)
    df = np.exp(-mkt.rate * (rem - t))
    return float(((cf.coupons + cf.redemption) @ df).mean())


def deltas(ts: TermSheet, mkt: mc.Market, t: float, perf_now: np.ndarray, missed: float, z: np.ndarray,
           bump: float = 0.01) -> np.ndarray:
    """dV / d(performance_i): index units to hold per 100 notional to offset the note's exposure to each index."""
    out = np.empty(len(perf_now))
    for i in range(len(perf_now)):
        up, dn = perf_now.copy(), perf_now.copy()
        up[i] *= 1 + bump
        dn[i] *= 1 - bump
        out[i] = (remaining_value(ts, mkt, t, up, missed, z) - remaining_value(ts, mkt, t, dn, missed, z)) / (
            2 * bump * perf_now[i])
    return out


@dataclass
class HedgeResult:
    hedged: np.ndarray            # issuer profit per scenario, present value per 100 notional
    unhedged: np.ndarray
    end_time: np.ndarray          # when the note ended (call date or maturity)
    worst_at_end: np.ndarray      # worst-of performance when it ended
    called: np.ndarray
    turnover: np.ndarray          # present value of index traded (buys + sells), per 100 notional
    rebalances_per_year: int
    inner_paths: int


def simulate(ts: TermSheet, mkt: mc.Market, n_scenarios: int = 300, rebalances_per_year: int = 12,
             inner_paths: int = 8_000, seed: int = 21, cost_bp: float = 0.0, hedge: bool = True,
             bump: float = 0.02) -> HedgeResult:
    """Issuer sells the note at its issue price and delta-hedges on a fixed grid. cost_bp: transaction cost in basis
    points of the value traded (the effect of any cost level can also be read off afterwards: profit - cost * turnover).
    hedge=False gives the unhedged book only."""
    rng = np.random.default_rng(seed)
    n_assets = len(mkt.vol)
    z_inner = mc.normals(inner_paths, ts.n_obs, n_assets, seed=seed + 1)
    L = np.linalg.cholesky(mkt.corr)
    grid = np.arange(1, int(round(ts.tenor_years * rebalances_per_year)) + 1) / rebalances_per_year
    obs = {round(t, 9): k for k, t in enumerate(ts.times)}
    dt = 1 / rebalances_per_year
    drift = (mkt.rate - mkt.div - 0.5 * mkt.vol ** 2) * dt
    cpn = ts.coupon_per_period * ts.notional
    h0 = deltas(ts, mkt, 0.0, np.ones(n_assets), 0.0, z_inner, bump) if hedge else np.zeros(n_assets)

    hedged = np.empty(n_scenarios)
    unhedged = np.empty(n_scenarios)
    end_time = np.empty(n_scenarios)
    worst_end = np.empty(n_scenarios)
    called_flag = np.zeros(n_scenarios, dtype=bool)
    turnover = np.zeros(n_scenarios)
    for s in range(n_scenarios):
        perf = np.ones(n_assets)
        missed = 0.0
        h = h0.copy()
        cash = ts.issue_price - h @ perf - cost_bp / 1e4 * np.abs(h) @ perf
        cash_u = ts.issue_price
        turnover[s] = np.abs(h) @ perf
        for t in grid:
            # move one rebalancing step: cash accrues, holdings earn dividends, indices move
            cash *= np.exp(mkt.rate * dt)
            cash_u *= np.exp(mkt.rate * dt)
            cash += (h * perf) @ (np.exp(mkt.div * dt) - 1)
            perf = perf * np.exp(drift + mkt.vol * np.sqrt(dt) * (L @ rng.standard_normal(n_assets)))
            k = obs.get(round(t, 9))
            done = False
            if k is not None:                          # observation date: coupon, call or maturity
                w = perf.min()
                pay = 0.0
                if w >= ts.coupon_barrier:
                    pay += cpn * (1 + (missed if ts.memory else 0))
                    missed = 0.0
                else:
                    missed += 1
                if k < ts.n_obs - 1 and w >= ts.autocall_trigger and k + 1 >= ts.first_call_obs:
                    pay += ts.notional
                    done, called_flag[s] = True, True
                elif k == ts.n_obs - 1:
                    pay += ts.notional * (w if w < ts.ki_barrier else 1.0)
                    done = True
                cash -= pay
                cash_u -= pay
            if done:
                cash += h @ perf - cost_bp / 1e4 * np.abs(h) @ perf          # unwind the hedge
                turnover[s] += np.abs(h) @ perf * np.exp(-mkt.rate * t)
                end_time[s], worst_end[s] = t, perf.min()
                break
            if hedge:
                h_new = deltas(ts, mkt, t, perf, missed, z_inner, bump)
                cash -= (h_new - h) @ perf + cost_bp / 1e4 * np.abs(h_new - h) @ perf
                turnover[s] += np.abs(h_new - h) @ perf * np.exp(-mkt.rate * t)
                h = h_new
        disc = np.exp(-mkt.rate * end_time[s])
        hedged[s] = cash * disc
        unhedged[s] = cash_u * disc
    return HedgeResult(hedged, unhedged, end_time, worst_end, called_flag, turnover, rebalances_per_year, inner_paths)
