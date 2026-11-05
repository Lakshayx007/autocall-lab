"""The note's terms, and the payoff applied to simulated paths of worst-of performance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TermSheet:
    """A worst-of autocallable note. Levels are fractions of the initial fixing (1.0 = 100%)."""
    tenor_years: float = 3.0
    obs_per_year: int = 4
    autocall_trigger: float = 1.00     # called if worst-of >= trigger on an observation date
    coupon_barrier: float = 0.70       # coupon paid if worst-of >= barrier on an observation date
    coupon_pa: float = 0.08            # coupon per year, paid per period (coupon_pa / obs_per_year)
    memory: bool = True                # missed coupons are paid when the coupon condition is met again
    ki_barrier: float = 0.60           # at maturity, if not called and worst-of < barrier: capital loss
    notional: float = 100.0
    issue_price: float = 100.0
    first_call_obs: int = 1            # first observation date on which the note can be called

    @property
    def n_obs(self) -> int:
        return int(round(self.tenor_years * self.obs_per_year))

    @property
    def times(self) -> np.ndarray:
        """Observation (and payment) times in years: 0.25, 0.5, ..., tenor."""
        return np.arange(1, self.n_obs + 1) / self.obs_per_year

    @property
    def coupon_per_period(self) -> float:
        return self.coupon_pa / self.obs_per_year


@dataclass
class Cashflows:
    """Undiscounted cash flows per path and observation date, split by type, plus how each path ended."""
    coupons: np.ndarray        # (paths, n_obs)
    coupon_units: np.ndarray   # (paths, n_obs): number of coupon periods paid on each date (memory can pay several)
    redemption: np.ndarray     # (paths, n_obs): notional at the call date, or the maturity redemption
    call_index: np.ndarray     # (paths,) observation index of the autocall, -1 if never called
    knocked_in: np.ndarray     # (paths,) True if not called and worst-of < ki barrier at maturity


def cashflows(worst: np.ndarray, ts: TermSheet, start: int = 0, missed0: np.ndarray | float = 0.0) -> Cashflows:
    """Apply the note's rules to worst-of performance paths (paths x dates, each value = min_i S_i(t) / S_i(0)).

    By default the paths cover all observation dates. To value the note part-way through its life, pass the remaining
    dates only, `start` = index of the first remaining date, and `missed0` = coupons already owed under the memory feature.
    """
    n_paths, n_obs = worst.shape
    last = ts.n_obs - 1
    cpn = ts.coupon_per_period * ts.notional
    coupons = np.zeros_like(worst)
    units = np.zeros_like(worst)
    redemption = np.zeros_like(worst)
    alive = np.ones(n_paths, dtype=bool)
    missed = np.zeros(n_paths) + missed0           # coupons owed under the memory feature
    call_index = np.full(n_paths, -1)
    for k in range(n_obs):
        w = worst[:, k]
        pay = alive & (w >= ts.coupon_barrier)
        units[pay, k] = 1 + (missed[pay] if ts.memory else 0)
        coupons[pay, k] = cpn * units[pay, k]
        missed[pay] = 0
        missed[alive & ~pay] += 1
        g = start + k                              # index of this date in the full schedule
        called = alive & (w >= ts.autocall_trigger) & (g + 1 >= ts.first_call_obs) & (g < last)
        redemption[called, k] = ts.notional
        call_index[called] = k
        alive &= ~called
    w_T = worst[:, -1]
    knocked_in = alive & (w_T < ts.ki_barrier)
    redemption[alive, -1] = np.where(knocked_in[alive], ts.notional * w_T[alive], ts.notional)
    return Cashflows(coupons, units, redemption, call_index, knocked_in)
