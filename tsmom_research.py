"""Public research API; implementation lives in focused ``tsmom`` modules."""

from tsmom import (
    Result,
    backtest,
    block_bootstrap,
    block_indices,
    cap_targets,
    comparison,
    full_strategy_bootstrap,
    performance,
    portfolio_vol_targets,
    targets_from_prices,
    walk_forward_recomputed,
)

__all__ = [
    "Result",
    "backtest",
    "block_bootstrap",
    "block_indices",
    "cap_targets",
    "comparison",
    "full_strategy_bootstrap",
    "performance",
    "portfolio_vol_targets",
    "targets_from_prices",
    "walk_forward_recomputed",
]
