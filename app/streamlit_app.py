"""Autocall Lab: price a worst-of autocallable note on the S&P 500, Euro Stoxx 50 and Nikkei 225.

Run locally:  streamlit run app/streamlit_app.py
Uses the market snapshot cached in data/, so every number can be reproduced from the repository.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autocall import market as M  # noqa: E402
from autocall import mc  # noqa: E402
from autocall import risk as R  # noqa: E402
from autocall.note import TermSheet, cashflows  # noqa: E402

ACCENT, GREY, LIGHT = "#1F4E79", "#7F8C9A", "#B8C2CC"
REPO = "https://github.com/Lakshayx007/autocall-lab"
SEED = 7
N_SHOW = 100            # sample paths drawn in the chart
VOL_METHODS = {"Historical, full sample": ("historical", None),
               "Historical, last 12 months": ("historical", 252),
               "EWMA (lambda 0.94)": ("ewma", None)}

st.set_page_config(page_title="Autocall Lab", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>""", unsafe_allow_html=True)


# ── cached computations ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_snapshot(method: str, window: int | None) -> M.MarketSnapshot:
    return M.snapshot(vol_method=method, window=window)


def build_market(snap: M.MarketSnapshot, vol_shift: float, corr_shift: float) -> mc.Market:
    mkt = mc.Market.from_snapshot(snap)
    return replace(mkt, vol=np.maximum(mkt.vol + vol_shift, 0.01), corr=R.shift_correlation(mkt.corr, corr_shift))


@st.cache_data(show_spinner=False)
def run_all(ts: TermSheet, method: str, window: int | None, vol_shift: float, corr_shift: float, n_paths: int,
            margin: float):
    snap = load_snapshot(method, window)
    mkt = build_market(snap, vol_shift, corr_shift)
    z = mc.normals(n_paths, ts.n_obs, len(snap.names), SEED)
    res = mc.price(ts, mkt, z=z)
    g = R.greeks(ts, mkt, snap.names, n_paths=n_paths, seed=SEED)
    try:
        fc = R.fair_coupon(ts, mkt, margin, z=z)
    except ValueError:                      # no coupon between 0% and 50% reaches the target price
        fc = float("nan")
    pick = np.linspace(0, n_paths - 1, N_SHOW).astype(int)       # spread across the set, both antithetic halves
    sample = mc.performance_paths(mkt, ts.times, z[pick]).min(axis=2)
    return res, g, fc, sample


# ── sidebar: inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Term sheet")
    coupon = st.slider("Coupon, % a year", 0.0, 20.0, 8.0, 0.1)
    tenor = st.select_slider("Tenor, years (quarterly observations)", [1, 2, 3, 4, 5], 3)
    trigger = st.slider("Autocall trigger, % of initial", 80, 110, 100, 5)
    coupon_barrier = st.slider("Coupon barrier, % of initial", 40, 100, 70, 5)
    memory = st.checkbox("Memory coupon (missed coupons paid later)", True)
    ki = st.slider("Capital barrier at maturity, % of initial", 30, 90, 60, 5)
    first_call = st.select_slider("First call date (quarter)", [1, 2, 3, 4], 1)

    st.header("Market")
    vol_label = st.selectbox("Volatility estimate", list(VOL_METHODS))
    vol_shift = st.slider("Shift all volatilities, points", -5.0, 5.0, 0.0, 0.5)
    corr_shift = st.slider("Shift all correlations", -0.3, 0.2, 0.0, 0.05)

    st.header("Structuring and simulation")
    margin = st.slider("Issuer margin for the fair coupon, points per 100", 0.0, 4.0, 2.0, 0.25)
    n_paths = st.select_slider("Monte Carlo paths", [20_000, 50_000, 100_000], 100_000,
                               format_func=lambda n: f"{n:,}")

ts = TermSheet(tenor_years=float(tenor), autocall_trigger=trigger / 100, coupon_barrier=coupon_barrier / 100,
               coupon_pa=coupon / 100, memory=memory, ki_barrier=ki / 100, first_call_obs=first_call)
method, window = VOL_METHODS[vol_label]

try:
    with st.spinner("Simulating..."):
        res, g, fc, sample = run_all(ts, method, window, vol_shift / 100, corr_shift, n_paths, margin)
except np.linalg.LinAlgError:
    st.error("That correlation shift gives an impossible correlation matrix. Choose a smaller shift.")
    st.stop()
snap = load_snapshot(method, window)
mkt = build_market(snap, vol_shift / 100, corr_shift)

# ── header ────────────────────────────────────────────────────────────────────
st.title("Autocall Lab")
st.markdown(
    f"Worst-of autocallable note on the **S&P 500, Euro Stoxx 50 and Nikkei 225**, priced by Monte Carlo. "
    f"Market data as of **{snap.as_of:%d %b %Y}** (fetched {snap.fetched_on}). Notional 100, issued at 100. "
    f"Method, validation and hedging analysis: [notebook and code on GitHub]({REPO}).")

# ── headline numbers ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fair value", f"{res.price:.2f}", help="Present value per 100 notional")
c1.caption(f"95% CI {res.ci95[0]:.2f} to {res.ci95[1]:.2f} (SE {res.stderr:.3f})")
c2.metric("Issuer margin at 100", f"{ts.issue_price - res.price:+.2f}")
c2.caption("Issue price minus fair value")
c3.metric("Called early", f"{1 - res.maturity_prob:.0%}")
c3.caption(f"Expected life {res.expected_life:.2f} years")
c4.metric("Capital loss", f"{res.loss_prob:.1%}")
c4.caption(f"Average repayment if so: {res.loss_given_loss:.0f}" if res.loss_prob > 0 else "No losses simulated")
c5.metric(f"Fair coupon, {margin:g}-pt margin", "n/a" if np.isnan(fc) else f"{fc:.2%}")
c5.caption("Coupon that sets fair value = 100 minus margin")

# ── charts ────────────────────────────────────────────────────────────────────
left, right = st.columns(2)
with left:
    times = np.concatenate([[0.0], ts.times])
    cf = cashflows(sample, ts)
    fig = go.Figure()
    for p in range(sample.shape[0]):
        end = cf.call_index[p] + 1 if cf.call_index[p] >= 0 else ts.n_obs
        y = np.concatenate([[1.0], sample[p, :end]]) * 100
        lost = cf.knocked_in[p]
        fig.add_trace(go.Scatter(x=times[:end + 1], y=y, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=ACCENT if lost else LIGHT, width=1.5 if lost else 0.8)))
    for lvl, name in ((trigger, "Autocall trigger"), (coupon_barrier, "Coupon barrier"), (ki, "Capital barrier")):
        fig.add_hline(y=lvl, line=dict(color=GREY, dash="dash", width=1), annotation_text=f"{name} {lvl}%",
                      annotation_position="right", annotation_font_size=11, annotation_font_color=GREY)
    n_lost = int(cf.knocked_in.sum())
    fig.update_layout(
        title=f"Most paths stop at an early call; {n_lost} of {sample.shape[0]} sample paths lose capital (dark)",
        xaxis_title="Years", yaxis_title="Worst index, % of initial", template="plotly_white", height=400,
        margin=dict(l=10, r=130, t=50, b=10), title_font_size=14)
    st.plotly_chart(fig, use_container_width=True)

with right:
    labels = [f"Q{k + 1}" for k in range(ts.n_obs - 1)] + ["Repaid", "Loss"]
    probs = list(res.call_prob[:-1]) + [res.maturity_prob - res.loss_prob, res.loss_prob]
    colours = [ACCENT] * (ts.n_obs - 1) + [GREY, "#2B2B2B"]
    k = int(np.argmax(probs))
    top = (f"an autocall at {labels[k]}" if k < ts.n_obs - 1 else
           "repayment of 100 at maturity" if k == ts.n_obs - 1 else "a capital loss")
    fig = go.Figure(go.Bar(x=labels, y=np.array(probs) * 100, marker_color=colours,
                           hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title=f"How the note ends: most likely outcome is {top}", xaxis_title="Outcome",
                      yaxis_title="Probability, %", template="plotly_white", height=400,
                      margin=dict(l=10, r=10, t=50, b=10), title_font_size=14, xaxis_tickangle=0)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Blue: autocall on each quarter. Grey: runs to maturity and repays 100. Black: capital loss at maturity.")

# ── risk and inputs ───────────────────────────────────────────────────────────
left, right = st.columns([3, 2])
with left:
    st.subheader("Sensitivities")
    d = g.loc[snap.names, "delta per 1% move"]
    st.caption(f"Bump and reprice on the same random numbers. The note moves most with the {d.idxmax()}, the index "
               f"most likely to end up the worst. Negative vega: the investor has sold protection against large falls.")
    show = pd.DataFrame({
        "Delta per 1% index move": g["delta per 1% move"].map(lambda x: f"{x:+.3f}"),
        "Exposure, % of notional": g["equivalent exposure, % of notional"].map(lambda x: f"{x:.1f}%"),
        "Vega per vol point": g["vega per vol point"].map(lambda x: f"{x:+.3f}"),
    })
    st.dataframe(show, use_container_width=True)

with right:
    st.subheader("Market inputs")
    st.caption(f"{vol_label}; weekly-return correlation; flat USD rate {mkt.rate:.2%} from 13-week T-bills.")
    inputs = pd.DataFrame({"Volatility": [f"{v:.1%}" for v in mkt.vol],
                           "Dividend yield": [f"{q:.2%}" for q in mkt.div]}, index=snap.names)
    st.dataframe(inputs, use_container_width=True)
    st.caption("Correlation of weekly returns")
    corr = pd.DataFrame(mkt.corr, index=snap.names, columns=snap.names).map(lambda x: f"{x:.2f}")
    st.dataframe(corr, use_container_width=True)

with st.expander("Model and assumptions"):
    st.markdown(f"""
- Correlated geometric Brownian motion under the risk-neutral measure, constant volatilities, drift = rate minus
  dividend yield. Simulated only on observation dates (exact steps), {n_paths:,} paths with antithetic variates, seed {SEED}.
- Volatility from daily index returns; correlation from weekly returns, because Tokyo, Europe and New York close at
  different times. A desk would use implied volatility surfaces and a smile model (local or stochastic volatility).
- Flat rate and dividend yields; no quanto adjustment for the euro and yen indices paid in USD; no issuer credit spread
  or funding benefit. Each would move the fair value and the fair coupon.
- Capital barrier observed at maturity only. Coupon, autocall and barrier tests use the worst of the three performances.
""")
