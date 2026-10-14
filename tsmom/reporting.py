"""Plots and notebook presentation, with no strategy logic in generated cells."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from jupyter_client import KernelManager
from nbclient import NotebookClient

from .config import DAILY_REFERENCE, TRADING_DAYS, ResearchConfig
from .experiments import ResearchRun


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_results(run: ResearchRun, bootstrap_paths: np.ndarray | None = None) -> None:
    output = run.config.output_dir
    figure, axis = plt.subplots(figsize=(11, 5))
    for name in (DAILY_REFERENCE, "Weekly next open", "Long only inverse vol", "SPY buy and hold"):
        returns = run.results[name].net
        equity = np.r_[1.0, np.cumprod(1 + returns.to_numpy())]
        dates = run.market.close.index[run.market.close.index.get_loc(run.start) - 1 :]
        axis.plot(dates, equity, label=name)
    axis.set(title="Matched dates: growth of $1 after trading fees", ylabel="Portfolio value")
    axis.legend()
    _save_figure(figure, output / "strategy_comparison.png")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    for axis, ensemble in zip(axes, ("Original", "Slower"), strict=True):
        for direction in ("long-only", "long/short"):
            returns = run.direction_results[f"{ensemble} {direction} momentum"].net
            (1 + returns).cumprod().plot(ax=axis, label=direction)
        axis.set(title=f"{ensemble} momentum: identical funding rates", ylabel="Growth of $1")
        axis.legend()
    _save_figure(figure, output / "funded_direction.png")

    figure, axis = plt.subplots(figsize=(11, 4))
    (1 + run.tables["07_walk_forward_returns"]).cumprod().plot(ax=axis)
    axis.set(title="Historical annual walk-forward tests", ylabel="Growth of $1")
    _save_figure(figure, output / "walk_forward.png")

    if bootstrap_paths is not None:
        figure, axis = plt.subplots(figsize=(11, 4))
        lower, median, upper = np.quantile(bootstrap_paths, [0.05, 0.5, 0.95], axis=0)
        years = np.arange(bootstrap_paths.shape[1]) / TRADING_DAYS
        axis.fill_between(years, lower, upper, alpha=0.25, label="Pointwise 5–95% bootstrap range")
        axis.plot(years, median, label="Median wealth")
        axis.axhline(1, color="gray", linewidth=0.8)
        axis.set(
            title="Daily strategy: conditional three-year paths, 63-session blocks",
            xlabel="Years",
            ylabel="Growth of $1",
        )
        axis.legend()
        _save_figure(figure, output / "bootstrap_paths.png")


NOTEBOOK_SECTIONS = (
    (
        "Daily versus weekly",
        "Keep the signal, sizing, dates, and costs fixed. Weekly trading uses the first available session of each ISO week. The legacy row uses idealized close fills and weight-difference turnover.",
        "01_daily_weekly",
        None,
    ),
    (
        "Rebalancing bands",
        "Bands are absolute portfolio-weight gaps, not price changes. Sign flips and exits override the band on scheduled trading days.",
        "02_rebalancing_bands",
        None,
    ),
    (
        "Signals and volatility",
        "Compare slower 63/126/252-session momentum, bounded continuous signals, and EWMA volatility independently against the original daily strategy.",
        "03_signal_volatility",
        None,
    ),
    (
        "Portfolio risk",
        "The 10% asset scaling parameter does not target 10% portfolio volatility. The portfolio variant uses trailing covariance, a forecast floor, and exposure caps. Caps constrain target weights; holdings can drift between trades.",
        "04_portfolio_risk",
        None,
    ),
    (
        "Benchmarks",
        "Isolate the effects of directional momentum, multiple horizons, and inverse-volatility sizing. The always-long benchmark differs from long-only momentum, which moves negative signals to cash.",
        "05_benchmarks",
        "strategy_comparison.png",
    ),
    (
        "Execution and funding",
        "Next-open execution separates overnight old-position returns from intraday new-position returns. Same-close execution is optimistic. Fixed funding rates are illustrative rather than historical broker quotes.",
        "06_execution_costs",
        None,
    ),
    (
        "Funded long-only versus long/short",
        "Both variants use 2% cash interest, 5% debit interest, and 1% short borrow. Excess Sharpe subtracts a fixed 2% risk-free rate. Negative weights are clipped to zero without reallocating the remaining weights. Restricted short proceeds earn no cash interest.",
        "06b_funded_long_short",
        "funded_direction.png",
    ),
    (
        "Historical walk-forward tests",
        "Select among five predefined rules on trailing past returns, then freeze the rule for the next year. Tests restart from cash and charge entry and liquidation fees. Retrospective design and prior inspection mean these are historical diagnostics, not untouched OOS evidence.",
        "07_walk_forward_performance",
        "walk_forward.png",
    ),
    (
        "Monte Carlo",
        "Resample paired 2020–2024 returns and fees using 20/63/126-session moving blocks, 5,000 paths, and a three-year horizon. Probabilities depend on the historical sample and stationarity. Percentile Sharpe ranges are descriptive, not multiple-testing-adjusted significance tests.",
        "08_monte_carlo_summary",
        "bootstrap_paths.png",
    ),
    (
        "Full-strategy simulation",
        "Jointly resample six-asset return rows and recalculate the strategy with 253 warm-up returns. Close-only synthetic prices use next-close execution. The 250 paths per block length are exploratory; block boundaries disrupt trends.",
        "09_full_rule_mc_summary",
        None,
    ),
    (
        "All fixed experiments",
        "Full-sample rankings are descriptive. Zero-RF comparisons below omit financing; the funded direction table above uses a different Sharpe definition. No historical winner is automatically promoted as a live strategy.",
        "10_all_comparisons",
        None,
    ),
)


def write_notebook(config: ResearchConfig, *, execute: bool = True) -> None:
    """Create a concise report; calculations remain in importable Python modules."""
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Multi-Asset Momentum Research\n\n"
            "I investigated time-series momentum after observing that an earlier cross-sectional "
            "mean-reversion signal degraded during trending regimes. This project compares "
            "trend direction, volatility sizing, trading frequency, and shorting across SPY, "
            "TLT, GLD, USO, DBA, and UUP. Prices span 2010–2024; common evaluation dates are "
            "2011-01-04 through 2024-12-31 after warm-up.\n\n"
            "Run `python run_tsmom_research.py` to reproduce the calculations and this report. "
            "The cells below display the exported results; set `RECOMPUTE = True` to run "
            "the complete research pipeline from the notebook."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "from tsmom.config import ResearchConfig\n"
            "from tsmom.pipeline import run_research\n\n"
            "ROOT = Path.cwd()\n"
            "OUTPUT = ROOT / 'tsmom_results'\n"
            "RECOMPUTE = False\n"
            "if RECOMPUTE or not (OUTPUT / '10_all_comparisons.csv').exists():\n"
            "    run_research(ResearchConfig(root=ROOT))\n\n"
            "pd.set_option('display.max_columns', 20)\n"
            "def show_table(name, indexed=True):\n"
            "    table = pd.read_csv(OUTPUT / f'{name}.csv', index_col=0 if indexed else None)\n"
            "    display(table.round(4))\n"
        ),
    ]
    unindexed = {"08_monte_carlo_summary", "09_full_rule_mc_summary"}
    for title, description, table, figure in NOTEBOOK_SECTIONS:
        source = f"show_table('{table}', indexed={table not in unindexed})"
        if figure:
            source += f"\ndisplay(Image(filename=str(OUTPUT / '{figure}')))"
        if table == "06b_funded_long_short":
            source += "\nshow_table('06b_shorting_increment')\nshow_table('06b_borrow_sensitivity', indexed=False)"
        if table == "07_walk_forward_performance":
            source += "\nshow_table('07_walk_forward_choices')"
        if table == "08_monte_carlo_summary":
            source += "\nshow_table('08_sharpe_uncertainty', indexed=False)"
        cells.extend(
            [
                nbformat.v4.new_markdown_cell(f"## {title}\n\n{description}"),
                nbformat.v4.new_code_cell(source),
            ]
        )
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
    )
    path = config.root / "tsmom_strategy.ipynb"
    nbformat.write(notebook, path)
    if execute:
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=600,
            allow_errors=False,
            resources={"metadata": {"path": str(config.root)}},
        )
        client.execute()
        nbformat.write(notebook, path)
