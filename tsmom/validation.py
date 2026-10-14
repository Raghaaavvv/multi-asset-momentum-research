"""Historical walk-forward diagnostics and dependence-aware resampling."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .backtest import Result, backtest
from .config import ORIGINAL_LOOKBACKS, TRADING_DAYS
from .metrics import performance
from .signals import targets_from_prices


def _liquidated_returns(result: Result, costs: pd.Series) -> pd.Series:
    """Charge year-end exits without mutating the result's accounting ledger."""
    returns = result.net.copy()
    exit_cost = result.exposures.iloc[-1].abs().dot(costs.reindex(result.exposures.columns)) / 10000
    returns.iloc[-1] = (1 + returns.iloc[-1]) * (1 - exit_cost) - 1
    return returns


def walk_forward_recomputed(
    close: pd.DataFrame,
    opens: pd.DataFrame,
    target_map: Mapping[str, pd.DataFrame],
    costs: pd.Series,
    start: str | pd.Timestamp,
    candidate_specs: Mapping[str, dict],
    first_test_year: int = 2016,
    train_years: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Select on past Sharpe and test next year, including entry and exit fees.

    Training histories carry positions; annual tests restart from cash. The
    retrospectively defined candidates make this a historical diagnostic.
    """
    if not candidate_specs or train_years < 1:
        raise ValueError("Candidates and a positive training window are required.")
    histories = {
        name: backtest(
            close,
            opens,
            target_map[spec["target"]],
            costs,
            start,
            frequency=spec.get("frequency", "daily"),
            band=spec.get("band", 0.0),
        )
        for name, spec in candidate_specs.items()
    }
    selections, selected_parts, reference_parts = [], [], []
    for year in range(first_test_year, close.index[-1].year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        training_start = cutoff - pd.DateOffset(years=train_years)
        scores = {
            name: performance(
                result.net.loc[(result.net.index >= training_start) & (result.net.index < cutoff)]
            )["Net Sharpe"]
            for name, result in histories.items()
        }
        chosen = max(scores, key=scores.get)
        test_dates = close.index[(close.index >= cutoff) & (close.index.year == year)]
        if test_dates.empty:
            continue
        spec = candidate_specs[chosen]
        selected = backtest(
            close.loc[: test_dates[-1]],
            opens.loc[: test_dates[-1]],
            target_map[spec["target"]],
            costs,
            test_dates[0],
            frequency=spec.get("frequency", "daily"),
            band=spec.get("band", 0.0),
        )
        reference = backtest(
            close.loc[: test_dates[-1]],
            opens.loc[: test_dates[-1]],
            target_map["ensemble"],
            costs,
            test_dates[0],
        )
        selected_returns = _liquidated_returns(selected, costs)
        reference_returns = _liquidated_returns(reference, costs)
        selected_parts.append(selected_returns)
        reference_parts.append(reference_returns)
        selections.append(
            {
                "Test year": year,
                "Training start": max(training_start, pd.Timestamp(start)),
                "Training end": histories[chosen].net.index[histories[chosen].net.index < cutoff][
                    -1
                ],
                "Selected": chosen,
                "Training Sharpe": scores[chosen],
                "Test Sharpe": performance(selected_returns)["Net Sharpe"],
                "Daily test Sharpe": performance(reference_returns)["Net Sharpe"],
            }
        )
    if not selected_parts:
        raise ValueError("No annual test periods are available.")
    return (
        pd.DataFrame(selections).set_index("Test year"),
        pd.concat(selected_parts),
        pd.concat(reference_parts),
    )


def block_indices(
    n: int,
    horizon: int,
    simulations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample non-circular moving blocks; joint columns share the same indices."""
    if min(n, horizon, simulations, block_length) < 1 or block_length > n:
        raise ValueError("Positive dimensions and a block length fitting the sample are required.")
    blocks = int(np.ceil(horizon / block_length))
    starts = rng.integers(0, n - block_length + 1, size=(simulations, blocks))
    indices = starts[..., None] + np.arange(block_length)
    return indices.reshape(simulations, -1)[:, :horizon]


def block_bootstrap(
    gross: pd.Series,
    costs: pd.Series,
    block_length: int = 63,
    simulations: int = 5000,
    horizon: int = 756,
    cost_multiplier: float = 1.0,
    seed: int = 20261014,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Resample paired returns and fees, sharing draws across fee stress scenarios.

    Fee stresses use the observed holding path, not newly simulated accounting.
    Probabilities are conditional on the sample and stationarity assumptions.
    """
    source = pd.concat([gross.rename("gross"), costs.rename("cost")], axis=1).dropna()
    if (
        not np.isfinite(source.to_numpy()).all()
        or (source["cost"] < 0).any()
        or cost_multiplier < 0
    ):
        raise ValueError("Finite returns and nonnegative fees are required.")
    indices = block_indices(
        len(source), horizon, simulations, block_length, np.random.default_rng(seed)
    )
    returns = (
        source["gross"].to_numpy()[indices] - cost_multiplier * source["cost"].to_numpy()[indices]
    )
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Invalid simulated returns.")
    equity = np.column_stack([np.ones(simulations), np.cumprod(1 + returns, axis=1)])
    drawdown = equity / np.maximum.accumulate(equity, axis=1) - 1
    longest = np.zeros(simulations, dtype=int)
    current = np.zeros(simulations, dtype=int)
    for underwater in (drawdown[:, 1:] < -1e-12).T:
        current = np.where(underwater, current + 1, 0)
        longest = np.maximum(longest, current)
    standard_deviation = returns.std(axis=1, ddof=1)
    sharpe = np.divide(
        returns.mean(axis=1) * np.sqrt(TRADING_DAYS),
        standard_deviation,
        out=np.full(simulations, np.nan),
        where=standard_deviation > 0,
    )
    metrics = pd.DataFrame(
        {
            "Terminal wealth": equity[:, -1],
            "CAGR": equity[:, -1] ** (TRADING_DAYS / horizon) - 1,
            "Max drawdown": drawdown.min(axis=1),
            "Longest drawdown (sessions)": longest,
            "Sharpe (zero RF)": sharpe,
        }
    )
    return metrics, equity


def full_strategy_bootstrap(
    close: pd.DataFrame,
    cost_bps: pd.Series,
    simulations: int = 250,
    horizon: int = 756,
    block_length: int = 252,
    seed: int = 20261014,
) -> pd.DataFrame:
    """Jointly resample asset returns and rerun sizing, holdings, and trading costs.

    Close-only synthetic paths use next-close execution. Block boundaries
    interrupt trends, so this is model sensitivity rather than a crisis forecast.
    """
    source = close.pct_change(fill_method=None).dropna().to_numpy()
    warmup = max(ORIGINAL_LOOKBACKS) + 1
    indices = block_indices(
        len(source),
        horizon + warmup,
        simulations,
        block_length,
        np.random.default_rng(seed),
    )
    dates = pd.bdate_range("2000-01-03", periods=horizon + warmup + 1)
    rows = []
    for draw in indices:
        prices = (
            np.vstack(
                [
                    np.ones(source.shape[1]),
                    np.cumprod(1 + source[draw], axis=0),
                ]
            )
            * 100
        )
        synthetic = pd.DataFrame(prices, index=dates, columns=close.columns)
        result = backtest(
            synthetic,
            synthetic,
            targets_from_prices(synthetic),
            cost_bps,
            start=dates[warmup + 1],
            execution="next_close",
        )
        row = performance(result.net)
        row["Costs / year"] = result.costs.mean() * TRADING_DAYS
        rows.append(row)
    return pd.DataFrame(rows)
