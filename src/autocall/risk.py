"""Sensitivities of the note (bump and reprice on common random numbers) and the structurer's fair-coupon solve."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import mc
from .note import TermSheet


def _z(ts: TermSheet, mkt: mc.Market, n_paths: int, seed: int) -> np.ndarray:
    return mc.normals(n_paths, ts.n_obs, len(mkt.vol), seed)


def shift_correlation(corr: np.ndarray, shift: float) -> np.ndarray:
    """Add `shift` to every off-diagonal correlation (kept within [-0.99, 0.99]); fails if no longer positive definite."""
    c = np.clip(np.asarray(corr) + shift * (1 - np.eye(len(corr))), -0.99, 0.99)
    np.fill_diagonal(c, 1.0)
    np.linalg.cholesky(c)
    return c


def greeks(ts: TermSheet, mkt: mc.Market, names: list[str], n_paths: int = mc.DEFAULT_PATHS,
           seed: int = mc.DEFAULT_SEED, spot_bump: float = 0.01, vol_bump: float = 0.01,
           corr_bump: float = 0.10) -> pd.DataFrame:
    """Per index: delta (price change for a 1% index move), the equivalent index exposure in % of notional, and vega
    (price change for +1 volatility point). Central differences, same random numbers for every bump."""
    z = _z(ts, mkt, n_paths, seed)
    n = len(names)
    rows = []
    for i, name in enumerate(names):
        up, dn = np.ones(n), np.ones(n)
        up[i], dn[i] = 1 + spot_bump, 1 - spot_bump
        d = (mc.price(ts, mkt, z=z, spot_rel=up).price - mc.price(ts, mkt, z=z, spot_rel=dn).price) / 2
        vu, vd = mkt.vol.copy(), mkt.vol.copy()
        vu[i] += vol_bump
        vd[i] -= vol_bump
        v = (mc.price(ts, replace(mkt, vol=vu), z=z).price - mc.price(ts, replace(mkt, vol=vd), z=z).price) / 2
        rows.append({"index": name, "delta per 1% move": d, "equivalent exposure, % of notional": d / spot_bump,
                     "vega per vol point": v})
    out = pd.DataFrame(rows).set_index("index")
    out.loc["All three together"] = [
        (mc.price(ts, mkt, z=z, spot_rel=np.full(n, 1 + spot_bump)).price
         - mc.price(ts, mkt, z=z, spot_rel=np.full(n, 1 - spot_bump)).price) / 2,
        np.nan,
        (mc.price(ts, replace(mkt, vol=mkt.vol + vol_bump), z=z).price
         - mc.price(ts, replace(mkt, vol=mkt.vol - vol_bump), z=z).price) / 2,
    ]
    out.loc["All three together", "equivalent exposure, % of notional"] = out.loc["All three together",
                                                                                "delta per 1% move"] / spot_bump
    return out


def correlation_sensitivity(ts: TermSheet, mkt: mc.Market, shifts=(-0.3, -0.2, -0.1, 0.0, 0.1, 0.2),
                            n_paths: int = mc.DEFAULT_PATHS, seed: int = mc.DEFAULT_SEED) -> pd.Series:
    """Note value when every pairwise correlation is shifted by the same amount."""
    z = _z(ts, mkt, n_paths, seed)
    return pd.Series({s: mc.price(ts, replace(mkt, corr=shift_correlation(mkt.corr, s)), z=z).price for s in shifts},
                     name="fair value")


def fair_coupon(ts: TermSheet, mkt: mc.Market, margin: float, n_paths: int = mc.DEFAULT_PATHS,
                seed: int = mc.DEFAULT_SEED, z: np.ndarray | None = None) -> float:
    """Annual coupon at which fair value = issue price - margin (margin in price points per 100 notional)."""
    z = _z(ts, mkt, n_paths, seed) if z is None else z
    target = ts.issue_price - margin
    f = lambda c: mc.price(replace(ts, coupon_pa=c), mkt, z=z).price - target  # noqa: E731
    return float(brentq(f, 0.0, 0.5, xtol=1e-6))


def coupon_grid(ts: TermSheet, mkt: mc.Market, margin: float, field: str, values,
                n_paths: int = mc.DEFAULT_PATHS, seed: int = mc.DEFAULT_SEED) -> pd.Series:
    """Fair coupon as one term-sheet feature varies (e.g. field='ki_barrier')."""
    z = _z(ts, mkt, n_paths, seed)
    return pd.Series({v: fair_coupon(replace(ts, **{field: v}), mkt, margin, z=z) for v in values}, name="fair coupon")
