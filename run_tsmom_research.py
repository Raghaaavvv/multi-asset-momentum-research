"""Command-line entry point for reproducible momentum research."""

import argparse
import logging
from pathlib import Path

from tsmom.config import ResearchConfig
from tsmom.pipeline import run_research
from tsmom.reporting import write_notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download prices instead of using the local cache.",
    )
    parser.add_argument(
        "--no-notebook",
        action="store_true",
        help="Run calculations and plots without starting Jupyter.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = ResearchConfig(root=Path(__file__).resolve().parent)
    run = run_research(config, refresh_data=args.refresh_data)
    write_notebook(config, execute=not args.no_notebook)
    print("\nFunded direction comparison (excess Sharpe vs 2% RF):")
    print(
        run.tables["06b_funded_long_short"][
            ["Excess Sharpe vs 2% RF", "CAGR", "Max drawdown"]
        ].round(4)
    )
    print(f"\nResults saved to {config.output_dir}")


if __name__ == "__main__":
    main()
