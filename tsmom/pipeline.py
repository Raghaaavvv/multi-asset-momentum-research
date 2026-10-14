"""Orchestrate the recorded experiments without import-time side effects."""

import logging

from .config import ResearchConfig
from .data import load_market_data
from .experiments import (
    ResearchRun,
    create_run,
    run_direction_comparison,
    run_financing_scenarios,
    run_structural_experiments,
    run_walk_forward,
    write_research_config,
)
from .reporting import plot_results
from .simulation import run_simulations

LOGGER = logging.getLogger(__name__)


def run_research(config: ResearchConfig, *, refresh_data: bool = False) -> ResearchRun:
    market = load_market_data(config, refresh=refresh_data)
    run = create_run(config, market)
    LOGGER.info("Evaluation: %s to %s", run.start.date(), market.close.index[-1].date())
    run_structural_experiments(run)
    run_financing_scenarios(run)
    run_direction_comparison(run)
    run_walk_forward(run)
    write_research_config(run)
    paths = run_simulations(run)
    plot_results(run, paths)
    return run
