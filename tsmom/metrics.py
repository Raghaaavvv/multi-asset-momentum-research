"""Performance statistics with explicit risk-free assumptions."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .backtest import Result
from .config import TRADING_DAYS


def performance(returns: pd.Series, risk_free_annual: float = 0.0) -> dict[str, float]:
    """Annualize by 252 sessions and include initial NAV in the high-water mark."""
    observations = pd.Series(returns).dropna()
    if observations.empty or not np.isfinite(observations).all() or (observations <= -1).any():
        raise ValueError("Returns must be finite, nonempty, and greater than -100%.")
    if not np.isfinite(risk_free_annual) or risk_free_annual <= -1:
        raise ValueError("Annual risk-free rate must be finite and greater than -100%.")
    equity = np.r_[1.0, np.cumprod(1 + observations.to_numpy())]
    drawdown = equity / np.maximum.accumulate(equity) - 1
    daily_risk_free = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = observations - daily_risk_free
    standard_deviation = excess.std(ddof=1)
    return {
        "Net Sharpe": excess.mean() / standard_deviation * np.sqrt(TRADING_DAYS)
        if standard_deviation > 0
        else np.nan,
        "CAGR": equity[-1] ** (TRADING_DAYS / len(observations)) - 1,
        "Annual vol": observations.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "Max drawdown": drawdown.min(),
    }


def comparison(
    results: Mapping[str, Result],
    baseline: str,
    risk_free_annual: float = 0.0,
    test_start: str = "2020-01-01",
) -> pd.DataFrame:
    """Compare fixed experiments on aligned dates; historical splits are diagnostic."""
    reference_dates = results[baseline].net.index
    rows = {}
    for name, result in results.items():
        if not result.net.index.equals(reference_dates):
            raise ValueError("Comparisons require identical evaluation dates.")
        row = performance(result.net, risk_free_annual)
        row.update(
            {
                "Gross Sharpe": performance(result.gross, risk_free_annual)["Net Sharpe"],
                "Turnover / year": result.turnover.mean() * TRADING_DAYS,
                "Costs / year": result.costs.mean() * TRADING_DAYS,
                "Financing drag / year": result.financing.mean() * TRADING_DAYS,
                "Mean gross exposure": result.exposures.abs().sum(axis=1).mean(),
                "Historical test Sharpe": performance(
                    result.net.loc[test_start:], risk_free_annual
                )["Net Sharpe"],
            }
        )
        rows[name] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["Delta net Sharpe"] = table["Net Sharpe"] - table.loc[baseline, "Net Sharpe"]
    table["Improves full-sample Sharpe"] = table["Delta net Sharpe"] > 1e-10
    return table
