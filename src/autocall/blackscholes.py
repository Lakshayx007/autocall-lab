"""Black-Scholes prices for European options with a continuous dividend yield, and implied volatility.

Used to validate the Monte Carlo engine (it must reproduce these prices) and to read implied volatility from quotes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(S, K, T, r, q, vol):
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    return d1, d1 - vol * np.sqrt(T)


def call(S, K, T, r, q, vol):
    d1, d2 = _d1d2(S, K, T, r, q, vol)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put(S, K, T, r, q, vol):
    d1, d2 = _d1d2(S, K, T, r, q, vol)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def digital_call(S, K, T, r, q, vol):
    """Pays 1 at T if S_T > K."""
    _, d2 = _d1d2(S, K, T, r, q, vol)
    return np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r, q, is_call=True):
    f = call if is_call else put
    return brentq(lambda v: f(S, K, T, r, q, v) - price, 1e-4, 3.0)
