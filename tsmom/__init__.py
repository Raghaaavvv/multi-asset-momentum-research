"""Multi-asset momentum research: signals, execution, and validation."""

from .backtest import Result, backtest
from .metrics import comparison, performance
from .signals import cap_targets, portfolio_vol_targets, targets_from_prices
from .validation import (
    block_bootstrap,
    block_indices,
    full_strategy_bootstrap,
    walk_forward_recomputed,
)

__all__ = [
    "Result",
    "backtest",
    "comparison",
    "performance",
    "cap_targets",
    "portfolio_vol_targets",
    "targets_from_prices",
    "block_bootstrap",
    "block_indices",
    "full_strategy_bootstrap",
    "walk_forward_recomputed",
]
