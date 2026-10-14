"""Check economic behavior, rather than mirroring implementation details."""

import unittest

import numpy as np
import pandas as pd

from tsmom import (
    backtest,
    block_bootstrap,
    block_indices,
    performance,
    portfolio_vol_targets,
    targets_from_prices,
    walk_forward_recomputed,
)


class ResearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2024-01-01", periods=5)
        self.close = pd.DataFrame({"A": [100.0, 110.0, 121.0, 121.0, 121.0]}, index=self.dates)
        self.opens = self.close.shift(1).fillna(self.close.iloc[0])
        self.targets = pd.DataFrame({"A": 0.5}, index=self.dates)
        self.costs = pd.Series({"A": 0.0})

    def run_strategy(self, **settings):
        return backtest(self.close, self.opens, self.targets, self.costs, self.dates[1], **settings)

    def test_execution_does_not_award_unheld_overnight_returns(self) -> None:
        self.opens.iloc[1] = 110.0
        expected = {"next_open": 0.0, "next_close": 0.0, "same_close": 0.05}
        for convention, first_return in expected.items():
            with self.subTest(execution=convention):
                result = self.run_strategy(execution=convention)
                self.assertAlmostEqual(result.net.iloc[0], first_return)
        delayed = self.run_strategy(execution="next_close")
        self.assertAlmostEqual(delayed.net.iloc[1], 0.05)

    def test_notional_costs_include_entry_and_both_sides_of_a_flip(self) -> None:
        self.close[:] = self.opens[:] = 100.0
        self.costs["A"] = 100.0
        self.targets.iloc[1:] = -0.5
        result = self.run_strategy(band=0.1)
        self.assertAlmostEqual(result.costs.iloc[0], 0.005)
        self.assertAlmostEqual(result.net.iloc[0], -0.005)
        self.assertAlmostEqual(result.turnover.iloc[1], 0.9975 / 0.995)
        self.assertAlmostEqual(result.costs.iloc[1], 0.009975 / 0.995)
        np.testing.assert_allclose(result.gross - result.costs, result.net)

    def test_schedules_preserve_drift_and_handle_holiday_weeks(self) -> None:
        daily = self.run_strategy()
        weekly = self.run_strategy(frequency="weekly")
        self.assertAlmostEqual((1 + daily.net.iloc[:2]).prod(), 1.1025)
        self.assertAlmostEqual((1 + weekly.net.iloc[:2]).prod(), 1.105)
        self.assertEqual(weekly.turnover.iloc[1], 0.0)
        held = self.run_strategy(frequency="buy_hold")
        self.assertTrue((held.turnover.iloc[1:] == 0).all())
        holiday_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-09"])
        prices = pd.DataFrame({"A": 100.0}, index=holiday_dates)
        targets = pd.DataFrame({"A": [0.5, 0.5, -0.5, -0.5]}, index=holiday_dates)
        holiday = backtest(
            prices, prices, targets, self.costs, holiday_dates[1], frequency="weekly"
        )
        np.testing.assert_allclose(holiday.turnover, [0.5, 0.0, 1.0])

    def test_funding_restricts_short_proceeds_and_charges_debt(self) -> None:
        self.close[:] = self.opens[:] = 100.0
        self.targets[:] = -0.5
        short = self.run_strategy(cash_rate=0.02, short_borrow_rate=0.01)
        self.assertAlmostEqual(short.cash_interest.iloc[1], 0.02 / 365.25)
        self.assertGreater(short.short_borrow.iloc[1], 0.0)
        np.testing.assert_allclose(short.financing, short.short_borrow - short.cash_interest)
        self.targets[:] = 0.0
        cash = self.run_strategy(cash_rate=0.02, short_borrow_rate=0.01)
        self.assertTrue((cash.short_borrow == 0).all())
        self.assertTrue((cash.net > 0).all())
        self.targets[:] = 2.0
        leveraged = self.run_strategy(debit_rate=0.05)
        self.assertAlmostEqual(leveraged.net.iloc[1], -0.05 / 365.25)

    def test_signals_and_returns_do_not_use_future_information(self) -> None:
        original = self.run_strategy()
        self.targets.iloc[-1] = -100.0
        pd.testing.assert_series_equal(original.net, self.run_strategy().net)
        dates = pd.bdate_range("2020-01-01", periods=300)
        changes = 0.0005 + 0.005 * np.sin(np.arange(300) / 7)
        prices = pd.DataFrame({"A": 100 * np.cumprod(1 + changes)}, index=dates)
        changed = prices.copy()
        changed.iloc[-5:] *= 2
        for build in (
            targets_from_prices,
            lambda frame: portfolio_vol_targets(
                targets_from_prices(frame), frame.pct_change(fill_method=None)
            ),
        ):
            pd.testing.assert_frame_equal(build(prices).iloc[:-5], build(changed).iloc[:-5])

    def test_metrics_include_initial_loss_and_subtract_risk_free_return(self) -> None:
        returns = pd.Series([-0.1, 0.01, 0.02])
        self.assertAlmostEqual(performance(returns)["Max drawdown"], -0.1)
        self.assertLess(
            performance(returns, 0.02)["Net Sharpe"], performance(returns)["Net Sharpe"]
        )

    def test_bootstrap_preserves_blocks_and_pairs_cost_stresses(self) -> None:
        indices = block_indices(30, 12, 5, 4, np.random.default_rng(7))
        self.assertTrue((np.diff(indices.reshape(5, 3, 4), axis=2) == 1).all())
        dates = pd.bdate_range("2020-01-01", periods=100)
        gross = pd.Series(np.linspace(-0.01, 0.012, 100), index=dates)
        fees = pd.Series(0.001, index=dates)
        settings = {"simulations": 20, "horizon": 60, "block_length": 10}
        normal, _ = block_bootstrap(gross, fees, **settings)
        stressed, _ = block_bootstrap(gross, fees, cost_multiplier=2, **settings)
        self.assertTrue((stressed["Terminal wealth"] < normal["Terminal wealth"]).all())

    def test_walk_forward_selection_ignores_test_period_prices(self) -> None:
        dates = pd.bdate_range("2014-01-01", "2016-12-30")
        changes = 0.001 + 0.002 * np.sin(np.arange(len(dates)) / 5)
        prices = pd.DataFrame({"A": 100 * np.cumprod(1 + changes)}, index=dates)
        target_map = {
            "ensemble": pd.DataFrame({"A": 0.5}, index=dates),
            "short": pd.DataFrame({"A": -0.5}, index=dates),
        }
        candidates = {"Long": {"target": "ensemble"}, "Short": {"target": "short"}}
        original, _, _ = walk_forward_recomputed(
            prices,
            prices,
            target_map,
            self.costs,
            dates[1],
            candidates,
            first_test_year=2016,
        )
        changed = prices.copy()
        changed.loc["2016":] *= np.linspace(1, 0.5, len(changed.loc["2016":]))[:, None]
        altered, _, _ = walk_forward_recomputed(
            changed,
            changed,
            target_map,
            self.costs,
            dates[1],
            candidates,
            first_test_year=2016,
        )
        self.assertEqual(original.loc[2016, "Selected"], "Long")
        self.assertEqual(original.loc[2016, "Selected"], altered.loc[2016, "Selected"])
        self.assertEqual(
            original.loc[2016, "Training Sharpe"], altered.loc[2016, "Training Sharpe"]
        )


if __name__ == "__main__":
    unittest.main()
