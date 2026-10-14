"""Download, validate, and cache adjusted OHLC without publishing raw data."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from .config import ResearchConfig


@dataclass(frozen=True)
class MarketData:
    close: pd.DataFrame
    open: pd.DataFrame


def load_market_data(config: ResearchConfig, *, refresh: bool = False) -> MarketData:
    """Prefer a CSV cache; migrate the original local pickle once if available."""
    legacy_cache = config.root / "tsmom_market_data.pkl"
    if config.cache_path.exists() and not refresh:
        data = pd.read_csv(
            config.cache_path,
            header=[0, 1],
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
    elif legacy_cache.exists() and not refresh:
        data = pd.read_pickle(legacy_cache)
    else:
        data = yf.download(
            tickers=list(config.tickers),
            start=config.start_date,
            end=config.end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected a nonempty multi-asset OHLC download.")
    close = data["Close"].reindex(columns=config.tickers)
    opens = data["Open"].reindex(columns=config.tickers)
    if close.isna().any().any() or opens.isna().any().any():
        raise ValueError("Missing asset prices; repair the data before running experiments.")
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and sorted.")
    if close.index.min() < pd.Timestamp(config.start_date) or close.index.max() >= pd.Timestamp(
        config.end_date
    ):
        raise ValueError("Cached data fall outside the configured research period.")
    for prices in (close, opens):
        if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
            raise ValueError("Adjusted price proxies must be positive and finite.")

    config.cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve an existing cache byte-for-byte for stable data fingerprints.
    if refresh or not config.cache_path.exists():
        data.to_csv(config.cache_path, float_format="%.17g")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "Yahoo Finance via yfinance",
        "tickers": list(config.tickers),
        "start_inclusive": config.start_date,
        "end_exclusive": config.end_date,
        "auto_adjust": True,
        "yfinance_version": yf.__version__,
        "cache_format": "CSV",
        "sha256": hashlib.sha256(config.cache_path.read_bytes()).hexdigest(),
        "observations": len(close),
        "note": "Adjusted OHLC are total-return proxies, not historical executable quotes.",
    }
    (config.output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return MarketData(close=close.copy(), open=opens.copy())
