"""Momentum direction and risk sizing, using data available at each close."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .config import ASSET_VOL_TARGET, ORIGINAL_LOOKBACKS, TRADING_DAYS, VOLATILITY_WINDOW


def realized_volatility(
    close: pd.DataFrame,
    estimator: Literal["rolling", "ewma"] = "rolling",
) -> pd.DataFrame:
    returns = close.pct_change(fill_method=None)
    if estimator == "rolling":
        daily_vol = returns.rolling(VOLATILITY_WINDOW).std()
    elif estimator == "ewma":
        daily_vol = returns.ewm(
            span=VOLATILITY_WINDOW,
            min_periods=VOLATILITY_WINDOW,
            adjust=False,
        ).std()
    else:
        raise ValueError(f"Unknown volatility estimator: {estimator}")
    return daily_vol * np.sqrt(TRADING_DAYS)


def targets_from_prices(
    close: pd.DataFrame,
    lookbacks: Sequence[int] = ORIGINAL_LOOKBACKS,
    inverse_vol: bool = True,
    vol_floor: float | None = None,
    estimator: Literal["rolling", "ewma"] = "rolling",
    continuous: bool = False,
) -> pd.DataFrame:
    """Average momentum directions, then size each asset before equal allocation.

    The 10% asset scaling parameter is not a 10% portfolio volatility target.
    Execution lag belongs to the backtest, not to this signal calculation.
    """
    if not lookbacks or any(period < 1 for period in lookbacks):
        raise ValueError("Momentum lookbacks must be positive session counts.")
    volatility = realized_volatility(close, estimator)
    if vol_floor is not None:
        if vol_floor <= 0:
            raise ValueError("Volatility floor must be positive.")
        volatility = volatility.clip(lower=vol_floor)
    components = []
    for period in lookbacks:
        trailing_return = close.pct_change(period, fill_method=None)
        component = (
            np.tanh(trailing_return / (volatility * np.sqrt(period / TRADING_DAYS)))
            if continuous
            else np.sign(trailing_return)
        )
        components.append(component)
    signal = sum(components) / len(components)
    targets = signal * (ASSET_VOL_TARGET / volatility) if inverse_vol else signal
    return targets / close.shape[1]


def cap_targets(
    targets: pd.DataFrame,
    asset_cap: float = 0.4,
    gross_cap: float = 2.0,
) -> pd.DataFrame:
    """Cap decision-time weights; drifting holdings can exceed caps between trades."""
    if asset_cap <= 0 or gross_cap <= 0:
        raise ValueError("Exposure caps must be positive.")
    capped = targets.clip(-asset_cap, asset_cap)
    gross_exposure = capped.abs().sum(axis=1)
    multiplier = np.minimum(1.0, gross_cap / gross_exposure.clip(lower=1e-12))
    return capped.mul(multiplier, axis=0)


def portfolio_vol_targets(
    targets: pd.DataFrame,
    returns: pd.DataFrame,
    target: float = 0.10,
    forecast_floor: float = 0.03,
    max_multiplier: float = 3.0,
    shrinkage: float = 0.5,
) -> pd.DataFrame:
    """Scale with a trailing covariance estimate, then apply exposure caps."""
    if not 0 <= shrinkage <= 1 or min(target, forecast_floor, max_multiplier) <= 0:
        raise ValueError("Invalid portfolio volatility settings.")
    returns = returns.reindex(index=targets.index, columns=targets.columns)
    output = targets.copy() * np.nan
    return_values = returns.to_numpy()
    weight_values = targets.to_numpy()
    for position in range(VOLATILITY_WINDOW - 1, len(targets)):
        history = return_values[position - VOLATILITY_WINDOW + 1 : position + 1]
        weights = weight_values[position]
        if not np.isfinite(history).all() or not np.isfinite(weights).all():
            continue
        covariance = np.atleast_2d(np.cov(history.T, ddof=1)) * TRADING_DAYS
        covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
        forecast = np.sqrt(max(weights @ covariance @ weights, 0.0))
        multiplier = min(max_multiplier, target / max(forecast_floor, forecast))
        output.iloc[position] = weights * multiplier
    return cap_targets(output)
