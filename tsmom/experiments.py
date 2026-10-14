"""Fixed research experiments; strategy choices are separate from presentation."""

import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from .backtest import Result, backtest
from .config import DAILY_REFERENCE, SLOWER_LOOKBACKS, TRADING_DAYS, ResearchConfig
from .data import MarketData
from .metrics import comparison, performance
from .signals import cap_targets, portfolio_vol_targets, realized_volatility, targets_from_prices
from .validation import walk_forward_recomputed

LOGGER = logging.getLogger(__name__)
WALK_FORWARD_CANDIDATES = {
    "Daily": {"target": "ensemble"},
    "Weekly": {"target": "ensemble", "frequency": "weekly"},
    "Daily band 2.5%": {"target": "ensemble", "band": 0.025},
    "Slower daily": {"target": "slower"},
    "Slower weekly": {"target": "slower", "frequency": "weekly"},
}


@dataclass
class ResearchRun:
    """State for one dataset; each stage owns explicit inputs and output tables."""

    config: ResearchConfig
    market: MarketData
    targets: dict[str, pd.DataFrame]
    cost_bps: pd.Series
    start: pd.Timestamp
    results: dict[str, Result] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    direction_results: dict[str, Result] = field(default_factory=dict)

    def run(self, name: str, target: str, **settings) -> Result:
        result = backtest(
            self.market.close,
            self.market.open,
            self.targets[target],
            self.cost_bps,
            self.start,
            **settings,
        )
        self.results[name] = result
        return result

    def save_table(self, name: str, table: pd.DataFrame, *, index: bool = True) -> None:
        self.tables[name] = table
        table.to_csv(self.config.output_dir / f"{name}.csv", index=index)
        LOGGER.info("Saved %s", name)

    def compare(self, name: str, strategies: list[str]) -> None:
        table = comparison(
            {strategy: self.results[strategy] for strategy in strategies},
            DAILY_REFERENCE,
            test_start=self.config.historical_test_start,
        )
        self.save_table(name, table)


def create_run(config: ResearchConfig, market: MarketData) -> ResearchRun:
    original = targets_from_prices(market.close)
    valid_decisions = original.dropna()
    if valid_decisions.empty:
        raise ValueError("The dataset is too short for the momentum warm-up.")
    first_decision = market.close.index.get_loc(valid_decisions.index[0])
    start = market.close.index[first_decision + 1]
    costs = pd.Series(config.slippage_bps).reindex(market.close.columns) + config.commission_bps
    return ResearchRun(config, market, {"ensemble": original}, costs, start)


def _legacy_result(run: ResearchRun) -> Result:
    """Reproduce the original weight-difference model for historical comparison."""
    returns = run.market.close.pct_change(fill_method=None)
    weights = run.targets["ensemble"].shift(1)
    filled = weights.fillna(0.0)
    turnover = filled.diff().fillna(filled).abs()
    costs = turnover.mul(run.cost_bps, axis=1).sum(axis=1) / 10000
    gross = (returns * weights).sum(axis=1, min_count=len(weights.columns)).dropna()
    index = gross.index
    return Result(
        net=gross - costs.reindex(index),
        gross=gross,
        costs=costs.reindex(index),
        turnover=turnover.sum(axis=1).reindex(index),
        exposures=weights.reindex(index),
        financing=pd.Series(0.0, index=index),
    )


def run_structural_experiments(run: ResearchRun) -> None:
    """Change one component at a time, with funding initially disabled."""
    run.results["Legacy daily"] = _legacy_result(run)
    run.run(DAILY_REFERENCE, "ensemble")
    run.run("Weekly next open", "ensemble", frequency="weekly")
    run.compare("01_daily_weekly", list(run.results))

    band_names = []
    for frequency in ("daily", "weekly"):
        for band in (0.01, 0.025, 0.05):
            name = f"{frequency.title()} band {band:.1%}"
            run.run(name, "ensemble", frequency=frequency, band=band)
            band_names.append(name)
    run.compare("02_rebalancing_bands", [DAILY_REFERENCE, "Weekly next open", *band_names])

    signal_specs = {
        "Slower ensemble daily": ("slower", {"lookbacks": SLOWER_LOOKBACKS}),
        "Continuous ensemble daily": ("continuous", {"continuous": True}),
        "EWMA volatility daily": ("ewma", {"estimator": "ewma"}),
    }
    for name, (key, settings) in signal_specs.items():
        run.targets[key] = targets_from_prices(run.market.close, **settings)
        run.run(name, key)
    run.compare("03_signal_volatility", [DAILY_REFERENCE, *signal_specs])

    run.targets["capped"] = cap_targets(targets_from_prices(run.market.close, vol_floor=0.05))
    run.targets["portfolio"] = portfolio_vol_targets(
        run.targets["capped"],
        run.market.close.pct_change(fill_method=None),
    )
    run.run("Asset floor and caps", "capped")
    run.run("Portfolio vol daily", "portfolio")
    run.run("Portfolio vol weekly", "portfolio", frequency="weekly")
    run.compare(
        "04_portfolio_risk",
        [
            DAILY_REFERENCE,
            "Asset floor and caps",
            "Portfolio vol daily",
            "Portfolio vol weekly",
        ],
    )

    run.targets["single"] = targets_from_prices(run.market.close, lookbacks=(252,))
    run.targets["unscaled"] = targets_from_prices(run.market.close, inverse_vol=False)
    run.targets["long_only"] = (0.10 / realized_volatility(run.market.close)) / len(
        run.config.tickers
    )
    benchmarks = {
        "Single horizon inverse vol": "single",
        "Ensemble without vol sizing": "unscaled",
        "Long only inverse vol": "long_only",
    }
    for name, key in benchmarks.items():
        run.run(name, key)
    run.targets["spy"] = pd.DataFrame(
        0.0, index=run.market.close.index, columns=run.market.close.columns
    )
    run.targets["spy"]["SPY"] = 1.0
    run.run("SPY buy and hold", "spy", frequency="buy_hold")
    run.compare("05_benchmarks", [DAILY_REFERENCE, *benchmarks, "SPY buy and hold"])

    execution_names = []
    for execution in ("same_close", "next_close"):
        name = f"Execution {execution}"
        run.run(name, "ensemble", execution=execution)
        execution_names.append(name)
    for frequency in ("daily", "weekly"):
        name = f"{frequency.title()} double trading costs"
        run.run(name, "ensemble", frequency=frequency, cost_multiplier=2.0)
        execution_names.append(name)
    run.compare("06_execution_costs", [DAILY_REFERENCE, "Weekly next open", *execution_names])

    run.save_table("10_all_comparisons", comparison(run.results, DAILY_REFERENCE))
    run.save_table(
        "10_daily_net_returns",
        pd.DataFrame({name: result.net for name, result in run.results.items()}),
    )


def run_financing_scenarios(run: ResearchRun) -> None:
    rates = run.config.funding.backtest_kwargs()
    funded = {
        DAILY_REFERENCE: run.results[DAILY_REFERENCE],
        "Debit 5%, short borrow 1%, no cash interest": backtest(
            run.market.close,
            run.market.open,
            run.targets["ensemble"],
            run.cost_bps,
            run.start,
            **{**rates, "cash_rate": 0.0},
        ),
        "Cash 2%, debit 5%, short borrow 1%": backtest(
            run.market.close,
            run.market.open,
            run.targets["ensemble"],
            run.cost_bps,
            run.start,
            **rates,
        ),
    }
    table = comparison(funded, DAILY_REFERENCE)
    table["Excess Sharpe vs fixed 2% RF"] = [
        performance(result.net, run.config.funding.risk_free_rate)["Net Sharpe"]
        for result in funded.values()
    ]
    run.save_table("06_financing_scenarios", table)


def _direction_metrics(result: Result, risk_free: float) -> dict[str, float]:
    metrics = performance(result.net, risk_free)
    metrics["Excess Sharpe vs 2% RF"] = metrics.pop("Net Sharpe")
    metrics.update(
        {
            "Raw Sharpe (zero RF)": performance(result.net)["Net Sharpe"],
            "2020–2024 excess Sharpe": performance(result.net.loc["2020":], risk_free)[
                "Net Sharpe"
            ],
            "Trading fees / year": result.costs.mean() * TRADING_DAYS,
            "Short borrow / year": result.short_borrow.mean() * TRADING_DAYS,
            "Net cash interest / year": result.cash_interest.mean() * TRADING_DAYS,
            "Mean long exposure": result.exposures.clip(lower=0).sum(axis=1).mean(),
            "Mean short exposure": -result.exposures.clip(upper=0).sum(axis=1).mean(),
        }
    )
    return metrics


def run_direction_comparison(run: ResearchRun) -> None:
    """Clip shorts to cash without renormalizing positive target weights."""
    rates = run.config.funding.backtest_kwargs()
    risk_free = run.config.funding.risk_free_rate
    direction_targets = {"Always-long inverse vol": run.targets["long_only"]}
    for ensemble, key in (("Original", "ensemble"), ("Slower", "slower")):
        direction_targets[f"{ensemble} long/short momentum"] = run.targets[key]
        direction_targets[f"{ensemble} long-only momentum"] = run.targets[key].clip(lower=0)
    run.direction_results = {
        name: backtest(run.market.close, run.market.open, target, run.cost_bps, run.start, **rates)
        for name, target in direction_targets.items()
    }
    table = pd.DataFrame.from_dict(
        {
            name: _direction_metrics(result, risk_free)
            for name, result in run.direction_results.items()
        },
        orient="index",
    )
    run.save_table("06b_funded_long_short", table)
    run.save_table(
        "06b_direction_net_returns",
        pd.DataFrame({name: result.net for name, result in run.direction_results.items()}),
    )

    pair_rows, annual_rows, sensitivity_rows = [], [], []
    for ensemble in ("Original", "Slower"):
        long_only = run.direction_results[f"{ensemble} long-only momentum"]
        long_short = run.direction_results[f"{ensemble} long/short momentum"]
        long_only_sharpe = performance(long_only.net, risk_free)["Net Sharpe"]
        delta = performance(long_short.net, risk_free)["Net Sharpe"] - long_only_sharpe
        pair_rows.append(
            {
                "Ensemble": ensemble,
                "Long/short minus long-only excess Sharpe": delta,
                "Shorting improved historical excess Sharpe": delta > 0,
            }
        )
        for year in sorted(long_only.net.index.year.unique()):
            long_only_return = (1 + long_only.net.loc[long_only.net.index.year == year]).prod() - 1
            long_short_return = (
                1 + long_short.net.loc[long_short.net.index.year == year]
            ).prod() - 1
            annual_rows.append(
                {
                    "Ensemble": ensemble,
                    "Year": year,
                    "Long-only return": long_only_return,
                    "Long/short return": long_short_return,
                    "Long/short minus long-only return": long_short_return - long_only_return,
                }
            )
        for borrow_rate in (0.0, 0.01, 0.03):
            stressed = (
                long_short
                if borrow_rate == rates["short_borrow_rate"]
                else backtest(
                    run.market.close,
                    run.market.open,
                    direction_targets[f"{ensemble} long/short momentum"],
                    run.cost_bps,
                    run.start,
                    **{**rates, "short_borrow_rate": borrow_rate},
                )
            )
            short_sharpe = performance(stressed.net, risk_free)["Net Sharpe"]
            sensitivity_rows.append(
                {
                    "Ensemble": ensemble,
                    "Annual short-borrow rate": borrow_rate,
                    "Long-only excess Sharpe": long_only_sharpe,
                    "Long/short excess Sharpe": short_sharpe,
                    "Shorting excess Sharpe delta": short_sharpe - long_only_sharpe,
                }
            )
    run.save_table("06b_shorting_increment", pd.DataFrame(pair_rows).set_index("Ensemble"))
    run.save_table("06b_direction_annual_returns", pd.DataFrame(annual_rows), index=False)
    run.save_table("06b_borrow_sensitivity", pd.DataFrame(sensitivity_rows), index=False)
    direction_config = {
        "annual_funding_rates": rates,
        "risk_free_annual": risk_free,
        "execution": "next_open",
        "frequency": "daily",
        "long_only_rule": "clip negatives to zero; no renormalization",
        "borrow_sensitivity_rates": [0.0, 0.01, 0.03],
        "note": "Fixed illustrative rates, not historical realized broker rates.",
    }
    (run.config.output_dir / "06b_direction_config.json").write_text(
        json.dumps(direction_config, indent=2) + "\n"
    )


def run_walk_forward(run: ResearchRun) -> None:
    choices, selected, reference = walk_forward_recomputed(
        run.market.close,
        run.market.open,
        run.targets,
        run.cost_bps,
        run.start,
        WALK_FORWARD_CANDIDATES,
    )
    run.save_table("07_walk_forward_choices", choices)
    run.save_table(
        "07_walk_forward_performance",
        pd.DataFrame(
            {
                "Past-only selected": performance(selected),
                "Daily reference with same annual resets": performance(reference),
            }
        ).T,
    )
    run.save_table(
        "07_walk_forward_returns",
        pd.concat(
            [
                selected.rename("Selected"),
                reference.rename("Daily"),
            ],
            axis=1,
        ),
    )


def write_research_config(run: ResearchRun) -> None:
    config = {
        "walk_forward_candidates": WALK_FORWARD_CANDIDATES,
        "selection_metric": "past net Sharpe, zero RF",
        "train_years": 5,
        "first_test_year": 2016,
        "baseline_execution": "next_open",
        "weekly_schedule": "first available session of each ISO week",
        "per_asset_one_way_cost_bps": run.cost_bps.to_dict(),
        "band_absolute_weight_gaps": [0.01, 0.025, 0.05],
        "portfolio_risk": {
            "annual_target": 0.10,
            "covariance_window": 63,
            "diagonal_shrinkage": 0.5,
            "forecast_floor": 0.03,
            "max_scaling_multiplier": 3.0,
            "asset_cap": 0.4,
            "gross_cap": 2.0,
        },
        "bootstrap": {
            "seed": run.config.seed,
            "pnl_sample": "2020-2024",
            "paths": run.config.bootstrap_paths,
            "horizon": run.config.simulation_horizon,
            "block_lengths": [20, 63, 126],
            "full_rule_paths_per_block_length": run.config.full_strategy_paths,
            "full_rule_block_lengths": [126, 252],
        },
        "multiple_testing_note": "All fixed variants are recorded; full-sample winners are descriptive.",
    }
    (run.config.output_dir / "research_config.json").write_text(json.dumps(config, indent=2) + "\n")
