"""Explicit, shared assumptions for the recorded research experiments."""

from dataclasses import dataclass, field
from pathlib import Path

TRADING_DAYS = 252
CALENDAR_DAYS = 365.25
VOLATILITY_WINDOW = 63
ASSET_VOL_TARGET = 0.10
ORIGINAL_LOOKBACKS = (21, 63, 126, 252)
SLOWER_LOOKBACKS = (63, 126, 252)
DAILY_REFERENCE = "Daily next open"


@dataclass(frozen=True)
class Funding:
    """Fixed annual scenario rates, not historical broker rates."""

    cash_rate: float = 0.02
    debit_rate: float = 0.05
    short_borrow_rate: float = 0.01
    risk_free_rate: float = 0.02

    def backtest_kwargs(self) -> dict[str, float]:
        return {
            "cash_rate": self.cash_rate,
            "debit_rate": self.debit_rate,
            "short_borrow_rate": self.short_borrow_rate,
        }


@dataclass(frozen=True)
class ResearchConfig:
    root: Path
    tickers: tuple[str, ...] = ("SPY", "TLT", "GLD", "USO", "DBA", "UUP")
    start_date: str = "2010-01-01"
    end_date: str = "2025-01-01"
    historical_test_start: str = "2020-01-01"
    commission_bps: float = 0.5
    slippage_bps: dict[str, float] = field(
        default_factory=lambda: {
            "SPY": 1.0,
            "TLT": 1.5,
            "GLD": 1.5,
            "USO": 5.0,
            "DBA": 5.0,
            "UUP": 2.0,
        }
    )
    funding: Funding = field(default_factory=Funding)
    seed: int = 20261014
    bootstrap_paths: int = 5000
    full_strategy_paths: int = 250
    simulation_horizon: int = 756

    @property
    def output_dir(self) -> Path:
        return self.root / "tsmom_results"

    @property
    def cache_path(self) -> Path:
        return self.root / ".cache" / "adjusted_ohlc.csv"
