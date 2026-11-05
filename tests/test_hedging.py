"""The issuer's delta hedge: revaluation part-way through life, and the effect of hedging on profit and loss."""
from dataclasses import replace

import numpy as np
import pytest

from autocall import hedging as H
from autocall import mc
from autocall.note import TermSheet

TS = replace(TermSheet(), coupon_pa=0.07)
MKT = mc.Market(vol=np.array([0.15, 0.20, 0.30]), corr=np.array([[1, .7, .6], [.7, 1, .6], [.6, .6, 1]]),
                rate=0.04, div=np.array([0.01, 0.028, 0.012]))


def test_revaluation_at_issue_matches_the_pricer():
    z = mc.normals(40_000, TS.n_obs, 3)
    v = H.remaining_value(TS, MKT, 0.0, np.ones(3), 0.0, z)
    assert v == pytest.approx(mc.price(TS, MKT, z=z).price, abs=1e-9)


def test_coupons_owed_add_value_part_way_through():
    z = mc.normals(20_000, TS.n_obs, 3)
    perf = np.array([0.9, 0.8, 0.75])
    owed = H.remaining_value(TS, MKT, 0.6, perf, 2.0, z)
    none = H.remaining_value(TS, MKT, 0.6, perf, 0.0, z)
    assert owed > none


def test_hedge_deltas_are_positive_at_issue():
    z = mc.normals(20_000, TS.n_obs, 3)
    assert (H.deltas(TS, MKT, 0.0, np.ones(3), 0.0, z, bump=0.02) > 0).all()


@pytest.fixture(scope="module")
def run():
    return H.simulate(TS, MKT, n_scenarios=60, rebalances_per_year=4, inner_paths=4_000, seed=3)


def test_hedging_cuts_the_spread_of_profit(run):
    assert run.hedged.std() < 0.6 * run.unhedged.std()


def test_unhedged_and_hedged_books_see_the_same_note(run):
    assert run.turnover.min() > 0
    assert np.isin(run.end_time, TS.times).all()
    assert run.called.mean() > 0.3
