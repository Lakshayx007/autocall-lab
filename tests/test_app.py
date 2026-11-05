"""Smoke test: the Streamlit app runs end to end and shows the same price as the pricer."""
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from autocall import market as M
from autocall import mc
from autocall.note import TermSheet

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_runs_and_matches_the_pricer():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    fair = next(m for m in at.metric if m.label == "Fair value")
    expected = mc.price(TermSheet(), mc.Market.from_snapshot(M.snapshot())).price
    assert float(fair.value) == round(expected, 2)


def test_app_reacts_to_a_lower_coupon():
    at = AppTest.from_file(APP, default_timeout=120).run()
    before = float(next(m for m in at.metric if m.label == "Fair value").value)
    at.slider[0].set_value(4.0).run()
    assert not at.exception
    after = float(next(m for m in at.metric if m.label == "Fair value").value)
    assert after < before and np.isfinite(after)
