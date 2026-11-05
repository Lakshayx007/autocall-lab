"""
Market data for the note: index closes, the US short rate, dividend yields, volatilities and correlations.

Everything is fetched from Yahoo Finance with yfinance and cached in data/, so the notebook and the app can run
without a network connection and always state the date their data is as of.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
INDICES = {"S&P 500": "^GSPC", "Euro Stoxx 50": "^STOXX50E", "Nikkei 225": "^N225"}
RATE_TICKER = "^IRX"                       # 13-week US Treasury bill yield, % per year
DIVIDEND_PROXIES = {"S&P 500": "SPY", "Euro Stoxx 50": "FEZ", "Nikkei 225": "1321.T"}   # index-tracking ETFs
TRADING_DAYS = 252
EWMA_LAMBDA = 0.94                         # RiskMetrics daily decay


# ── fetching and caching ─────────────────────────────────────────────────────
def fetch(start: str = "2021-10-01", end: str | None = None) -> None:
    """Download closes, the T-bill yield and ETF dividend yields; write data/prices.csv and data/market_meta.json."""
    import yfinance as yf

    tickers = list(INDICES.values()) + [RATE_TICKER]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Close"]
    prices = raw.rename(columns={v: k for k, v in INDICES.items()} | {RATE_TICKER: "T-bill yield %"})
    prices.index = pd.to_datetime(prices.index).date
    prices.index.name = "date"
    DATA.mkdir(exist_ok=True)
    prices.to_csv(DATA / "prices.csv", float_format="%.6f")

    divs = {}
    for name, etf in DIVIDEND_PROXIES.items():
        info = yf.Ticker(etf).info
        divs[name] = {"etf": etf, "etf_name": info.get("longName"), "yield": float(info.get("yield") or 0.0)}
    meta = {"fetched_on": date.today().isoformat(), "source": "Yahoo Finance via yfinance",
            "indices": INDICES, "rate_ticker": RATE_TICKER, "dividend_yields": divs}
    spot = float(prices["S&P 500"].dropna().iloc[-1])
    r = float(np.log1p(prices["T-bill yield %"].dropna().iloc[-1] / 100))
    meta["spx_implied"] = spx_implied_vol(spot, r, divs["S&P 500"]["yield"], as_of=pd.Timestamp(prices.index[-1]))
    (DATA / "market_meta.json").write_text(json.dumps(meta, indent=2))


def spx_implied_vol(spot: float, r: float, q: float, as_of: pd.Timestamp, years: float = 3.0) -> dict | None:
    """At-the-money implied volatility of S&P 500 options expiring nearest to `years`, inverted with Black-Scholes from
    bid-ask mid prices of the three strikes nearest the spot (calls and puts, which should agree)."""
    import time

    import yfinance as yf

    from .blackscholes import implied_vol
    t = yf.Ticker("^SPX")
    expiries = ()
    for _ in range(4):                      # Yahoo sometimes rate-limits the first request
        expiries = t.options
        if expiries:
            break
        time.sleep(10)
    if not expiries:
        return None
    exp = min(expiries, key=lambda e: abs((pd.Timestamp(e) - as_of).days / 365.25 - years))
    T = (pd.Timestamp(exp) - as_of).days / 365.25
    chain = t.option_chain(exp)
    rows = []
    for kind, df in (("call", chain.calls), ("put", chain.puts)):
        df = df[(df.bid > 0) & (df.ask > 0)]
        for _, o in df.iloc[(df.strike - spot).abs().argsort()[:3]].iterrows():
            mid = (o.bid + o.ask) / 2
            rows.append({"type": kind, "strike": float(o.strike), "mid": float(mid),
                         "iv": float(implied_vol(mid, spot, o.strike, T, r, q, kind == "call"))})
    ivs = [x["iv"] for x in rows]
    return {"expiry": exp, "years": round(T, 3), "atm_iv": float(np.median(ivs)), "quotes": rows}


def load_prices() -> pd.DataFrame:
    """Cached closes on the union of the three markets' trading days (NaN where a market was closed)."""
    p = pd.read_csv(DATA / "prices.csv", index_col="date", parse_dates=["date"])
    return p


def load_meta() -> dict:
    return json.loads((DATA / "market_meta.json").read_text())


# ── calendars and returns ────────────────────────────────────────────────────
def common_closes(prices: pd.DataFrame) -> pd.DataFrame:
    """Index closes on days when all three markets traded (holidays differ between the US, Europe and Japan)."""
    return prices[list(INDICES)].dropna(how="any")


def log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns between consecutive common trading days."""
    return np.log(closes).diff().dropna()


def calendar_gaps(prices: pd.DataFrame) -> pd.DataFrame:
    """For each index: days it is missing on the union calendar, and days dropped by keeping common days only."""
    idx = prices[list(INDICES)]
    common = idx.dropna(how="any")
    return pd.DataFrame({"own trading days": idx.notna().sum(),
                         "missing on union calendar": idx.isna().sum(),
                         "days kept (common)": len(common)})


# ── volatility ───────────────────────────────────────────────────────────────
def historical_vol(rets: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Annualised standard deviation of daily log returns over the last `window` days (all days if None)."""
    r = rets if window is None else rets.iloc[-window:]
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS)


def ewma_vol(rets: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """RiskMetrics EWMA volatility path, annualised: var_t = lam * var_{t-1} + (1 - lam) * r_{t-1}^2."""
    var = np.empty_like(rets.values)
    var[0] = rets.iloc[:30].var().values                     # seed with the first month's sample variance
    for t in range(1, len(rets)):
        var[t] = lam * var[t - 1] + (1 - lam) * rets.values[t - 1] ** 2
    return pd.DataFrame(np.sqrt(var * TRADING_DAYS), index=rets.index, columns=rets.columns)


# ── correlation ──────────────────────────────────────────────────────────────
def correlation(rets: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Correlation of non-overlapping `horizon`-day log returns. Longer horizons soften the timing mismatch: Tokyo
    closes before Europe opens and Europe before New York closes, so same-day returns see different news."""
    if horizon == 1:
        return rets.corr()
    summed = rets.groupby(np.arange(len(rets)) // horizon).sum()
    summed = summed.iloc[:-1] if len(rets) % horizon else summed        # drop an incomplete last block
    return summed.corr()


def lead_lag_correlation(rets: pd.DataFrame, a: str, b: str) -> dict:
    """Same-day correlation of a and b, and with a's next-day return (does a react a day later to b's news?)."""
    return {"same day": rets[a].corr(rets[b]), f"{a} next day": rets[a].shift(-1).corr(rets[b])}


def is_positive_definite(corr: pd.DataFrame | np.ndarray) -> tuple[bool, np.ndarray]:
    eig = np.linalg.eigvalsh(np.asarray(corr))
    return bool(eig.min() > 0), eig


# ── rates and dividends ──────────────────────────────────────────────────────
def short_rate(prices: pd.DataFrame) -> tuple[float, pd.Timestamp]:
    """Latest 13-week T-bill yield as a continuously compounded annual rate."""
    s = prices["T-bill yield %"].dropna()
    return float(np.log1p(s.iloc[-1] / 100)), s.index[-1]


def dividend_yields(meta: dict) -> pd.Series:
    return pd.Series({k: v["yield"] for k, v in meta["dividend_yields"].items()})


@dataclass
class MarketSnapshot:
    """Everything the pricer needs, with the dates it is as of."""
    names: list[str]
    spot: np.ndarray
    vol: np.ndarray
    corr: np.ndarray
    rate: float
    div: np.ndarray
    as_of: pd.Timestamp
    fetched_on: str
    notes: dict = field(default_factory=dict)


def snapshot(vol_method: str = "historical", corr_horizon: int = 5, window: int | None = None) -> MarketSnapshot:
    """Market inputs as of the last common close.

    vol_method: 'historical' (annualised std of daily log returns over `window` days, all available days if None) or
    'ewma' (RiskMetrics, lambda 0.94). corr_horizon: length in days of the non-overlapping returns used for
    correlation (5 = weekly, which removes most of the closing-time mismatch between Tokyo, Europe and New York).
    """
    prices, meta = load_prices(), load_meta()
    closes = common_closes(prices)
    rets = log_returns(closes)
    vol = ewma_vol(rets).iloc[-1] if vol_method == "ewma" else historical_vol(rets, window)
    corr = correlation(rets if window is None else rets.iloc[-window:], corr_horizon)
    r, r_date = short_rate(prices)
    names = list(INDICES)
    return MarketSnapshot(names=names, spot=closes.iloc[-1][names].values.astype(float), vol=vol[names].values,
                          corr=corr.loc[names, names].values, rate=r, div=dividend_yields(meta)[names].values,
                          as_of=closes.index[-1], fetched_on=meta["fetched_on"],
                          notes={"vol_method": vol_method, "corr_horizon_days": corr_horizon,
                                 "window_days": window or len(rets), "rate_date": str(r_date.date())})
