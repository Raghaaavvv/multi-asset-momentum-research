"""Recorded Monte Carlo experiments and their exported summary statistics."""

import numpy as np
import pandas as pd

from .config import DAILY_REFERENCE
from .experiments import ResearchRun
from .metrics import performance
from .validation import block_bootstrap, full_strategy_bootstrap


def run_simulations(run: ResearchRun) -> np.ndarray:
    """Run fixed P&L and full-rule bootstrap experiments; return the plotted paths."""
    summary_rows, uncertainty_rows = [], []
    plotted_paths = None
    config = run.config
    for name in (DAILY_REFERENCE, "Weekly next open"):
        result = run.results[name]
        gross = result.gross.loc[config.historical_test_start :]
        fees = result.costs.loc[gross.index]
        for block_length in (20, 63, 126):
            for fee_multiplier in (1, 2):
                metrics, paths = block_bootstrap(
                    gross,
                    fees,
                    block_length=block_length,
                    cost_multiplier=fee_multiplier,
                    simulations=config.bootstrap_paths,
                    horizon=config.simulation_horizon,
                    seed=config.seed,
                )
                filename = f"08_mc_{name.split()[0].lower()}_{block_length}_{fee_multiplier}x"
                run.save_table(filename, metrics, index=False)
                summary_rows.append(
                    {
                        "Rule": name,
                        "Block sessions": block_length,
                        "Cost multiplier": fee_multiplier,
                        "Loss probability (3y)": (metrics["Terminal wealth"] < 1).mean(),
                        "Drawdown > 15% probability": (metrics["Max drawdown"] < -0.15).mean(),
                        "CAGR p05": metrics["CAGR"].quantile(0.05),
                        "CAGR median": metrics["CAGR"].median(),
                        "CAGR p95": metrics["CAGR"].quantile(0.95),
                        "Drawdown p05 (severe)": metrics["Max drawdown"].quantile(0.05),
                        "Drawdown median": metrics["Max drawdown"].median(),
                        "Longest drawdown p95 (sessions)": metrics[
                            "Longest drawdown (sessions)"
                        ].quantile(0.95),
                    }
                )
                if name == DAILY_REFERENCE and block_length == 63 and fee_multiplier == 1:
                    plotted_paths = paths

            uncertainty, _ = block_bootstrap(
                gross,
                fees,
                block_length=block_length,
                horizon=len(gross),
                simulations=config.bootstrap_paths,
                seed=config.seed,
            )
            quantiles = uncertainty["Sharpe (zero RF)"].quantile([0.025, 0.5, 0.975])
            uncertainty_rows.append(
                {
                    "Rule": name,
                    "Block sessions": block_length,
                    "Historical Sharpe": performance(result.net.loc[gross.index])["Net Sharpe"],
                    "Bootstrap Sharpe p025": quantiles.iloc[0],
                    "Bootstrap Sharpe median": quantiles.iloc[1],
                    "Bootstrap Sharpe p975": quantiles.iloc[2],
                }
            )
    run.save_table("08_monte_carlo_summary", pd.DataFrame(summary_rows), index=False)
    run.save_table("08_sharpe_uncertainty", pd.DataFrame(uncertainty_rows), index=False)

    full_rule_rows = []
    for block_length in (126, 252):
        metrics = full_strategy_bootstrap(
            run.market.close,
            run.cost_bps,
            simulations=config.full_strategy_paths,
            horizon=config.simulation_horizon,
            block_length=block_length,
            seed=config.seed,
        )
        run.save_table(f"09_full_rule_mc_{block_length}", metrics, index=False)
        full_rule_rows.append(
            {
                "Block sessions": block_length,
                "Paths": len(metrics),
                "Loss probability (3y)": (metrics["CAGR"] < 0).mean(),
                "CAGR p05": metrics["CAGR"].quantile(0.05),
                "CAGR median": metrics["CAGR"].median(),
                "Net Sharpe median": metrics["Net Sharpe"].median(),
                "Drawdown p05 (severe)": metrics["Max drawdown"].quantile(0.05),
                "Annual fees median": metrics["Costs / year"].median(),
            }
        )
    run.save_table("09_full_rule_mc_summary", pd.DataFrame(full_rule_rows), index=False)
    return plotted_paths
