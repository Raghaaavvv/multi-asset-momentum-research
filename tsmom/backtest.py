"""Dollar-holdings backtest with explicit execution and funding conventions."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import CALENDAR_DAYS

Frequency = Literal["daily", "weekly", "buy_hold"]
Execution = Literal["same_close", "next_open", "next_close"]


@dataclass(frozen=True)
class Result:
    """Aligned daily observations; monetary quantities are fractions of opening NAV.

    `gross` adds back trading fees on the same funded holding path. It is not an
    independently compounded frictionless strategy. `financing` is borrow fees
    minus net cash interest; positive values represent a drag on returns.
    """

    net: pd.Series
    gross: pd.Series
    costs: pd.Series
    turnover: pd.Series
    exposures: pd.DataFrame
    financing: pd.Series
    cash_interest: pd.Series | None = None
    short_borrow: pd.Series | None = None


def _rebalance(
    holdings: np.ndarray,
    cash: float,
    desired_weights: np.ndarray,
    cost_rates: np.ndarray,
    band: float,
) -> tuple[np.ndarray, float, float, float]:
    """Apply one-way fees to actual traded notional, using pre-fee NAV targets."""
    nav = cash + holdings.sum()
    if nav <= 0:
        raise RuntimeError("Portfolio equity exhausted before rebalancing.")
    actual_weights = holdings / nav
    trade_mask = np.abs(desired_weights - actual_weights) > band
    trade_mask |= desired_weights * actual_weights < 0
    trade_mask |= (desired_weights == 0) & (actual_weights != 0)
    changes = np.where(trade_mask, desired_weights * nav - holdings, 0.0)
    fee = float(np.abs(changes) @ cost_rates)
    traded_notional = float(np.abs(changes).sum())
    return holdings + changes, cash - changes.sum() - fee, fee, traded_notional


def backtest(
    close: pd.DataFrame,
    open_prices: pd.DataFrame,
    targets: pd.DataFrame,
    cost_bps: pd.Series,
    start: str | pd.Timestamp,
    frequency: Frequency = "daily",
    band: float = 0.0,
    execution: Execution = "next_open",
    cash_rate: float = 0.0,
    debit_rate: float = 0.0,
    short_borrow_rate: float = 0.0,
    cost_multiplier: float = 1.0,
) -> Result:
    """Carry holdings and cash between trades; execute previous-close targets.

    Weekly trades use the first available session of each ISO week. Sign flips
    and complete exits override the band on scheduled trading days. Funding
    accrues on previous-close balances using elapsed calendar days. Short-sale
    proceeds are restricted collateral and earn no cash interest. Adjusted
    prices already reflect distributions, including the short dividend burden.
    """
    if frequency not in {"daily", "weekly", "buy_hold"}:
        raise ValueError(f"Unknown rebalance frequency: {frequency}")
    if execution not in {"same_close", "next_open", "next_close"}:
        raise ValueError(f"Unknown execution convention: {execution}")
    settings = np.array([band, cash_rate, debit_rate, short_borrow_rate, cost_multiplier])
    if not np.isfinite(settings).all() or (settings < 0).any():
        raise ValueError(
            "Bands, funding rates, and cost multipliers must be finite and nonnegative."
        )
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and sorted.")
    opens = open_prices.reindex(index=close.index, columns=close.columns)
    targets = targets.reindex(index=close.index, columns=close.columns)
    close_values = close.to_numpy()
    open_values = opens.to_numpy()
    weight_values = targets.to_numpy()
    cost_rates = cost_bps.reindex(close.columns).to_numpy() * cost_multiplier / 10000
    for values in (close_values, open_values):
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Adjusted price proxies must be complete, positive, and finite.")
    if not np.isfinite(cost_rates).all() or (cost_rates < 0).any():
        raise ValueError("Every asset needs a finite, nonnegative one-way trading cost.")
    beginning = close.index.get_loc(pd.Timestamp(start))
    if beginning < 1 or not np.isfinite(weight_values[beginning - 1 : -1]).all():
        raise ValueError("A complete previous-close signal is required throughout evaluation.")

    holdings = np.zeros(close.shape[1])
    cash = 1.0
    last_week: tuple[int, int] | None = None
    ledger: list[dict[str, float]] = []
    exposure_history: list[np.ndarray] = []
    for position in range(beginning, len(close)):
        opening_nav = cash + holdings.sum()
        elapsed_years = (close.index[position] - close.index[position - 1]).days / CALENDAR_DAYS
        short_notional = np.maximum(-holdings, 0).sum()
        creditable_cash = max(cash - short_notional, 0)
        interest = (creditable_cash * cash_rate + min(cash, 0) * debit_rate) * elapsed_years
        borrow_fee = short_notional * short_borrow_rate * elapsed_years
        cash += interest - borrow_fee

        iso_date = close.index[position].isocalendar()
        week = (iso_date.year, iso_date.week)
        scheduled = (
            frequency == "daily"
            or (frequency == "weekly" and week != last_week)
            or (frequency == "buy_hold" and position == beginning)
        )
        last_week = week
        fee = traded_notional = 0.0
        if execution == "next_open":
            holdings = holdings * open_values[position] / close_values[position - 1]
        elif execution == "next_close":
            holdings = holdings * close_values[position] / close_values[position - 1]

        if scheduled:
            holdings, cash, fee, traded_notional = _rebalance(
                holdings,
                cash,
                weight_values[position - 1],
                cost_rates,
                band,
            )

        if execution == "next_open":
            holdings = holdings * close_values[position] / open_values[position]
        elif execution == "same_close":
            # Optimistic: the previous close supplies both the signal and fill.
            holdings = holdings * close_values[position] / close_values[position - 1]

        closing_nav = cash + holdings.sum()
        if not np.isfinite(closing_nav) or closing_nav <= 0:
            raise RuntimeError("Non-positive or non-finite closing portfolio equity.")
        ledger.append(
            {
                "net": closing_nav / opening_nav - 1,
                "gross": (closing_nav + fee) / opening_nav - 1,
                "costs": fee / opening_nav,
                "turnover": traded_notional / opening_nav,
                "financing": (borrow_fee - interest) / opening_nav,
                "cash_interest": interest / opening_nav,
                "short_borrow": borrow_fee / opening_nav,
            }
        )
        exposure_history.append(holdings / closing_nav)

    index = close.index[beginning:]
    daily = pd.DataFrame(ledger, index=index)
    return Result(
        net=daily["net"],
        gross=daily["gross"],
        costs=daily["costs"],
        turnover=daily["turnover"],
        financing=daily["financing"],
        exposures=pd.DataFrame(exposure_history, index=index, columns=close.columns),
        cash_interest=daily["cash_interest"],
        short_borrow=daily["short_borrow"],
    )
