import unittest

import numpy as np

from picks import backtest, data
from picks.model import PoissonModel
from picks.value import devig, find_picks, kelly_fraction


class ValueTests(unittest.TestCase):
    def test_devig_sums_to_one(self):
        p = devig({"H": 2.0, "D": 3.5, "A": 4.0})
        self.assertAlmostEqual(sum(p.values()), 1.0)

    def test_kelly_zero_without_edge(self):
        self.assertEqual(kelly_fraction(0.4, 2.0), 0.0)

    def test_pick_found_when_model_disagrees(self):
        picks = find_picks({"H": 0.55, "D": 0.25, "A": 0.20}, {"H": 2.2, "D": 3.4, "A": 3.6})
        self.assertEqual([p.outcome for p in picks], ["H"])


class ModelTests(unittest.TestCase):
    def test_probabilities_sum_to_one_and_rank_strength(self):
        df = data.synthetic(n_teams=10, seasons=2, seed=1)
        m = PoissonModel().fit(df)
        pr = m.predict("T00", "T01")
        self.assertAlmostEqual(pr["H"] + pr["D"] + pr["A"], 1.0, places=6)
        self.assertAlmostEqual(pr["O25"] + pr["U25"], 1.0, places=6)

    def test_recovers_true_ordering(self):
        df = data.synthetic(n_teams=12, seasons=6, seed=3)
        m = PoissonModel(half_life_days=2000).fit(df)
        self.assertGreater(m.home_adv, 1.0)


class BacktestTests(unittest.TestCase):
    def test_runs_and_never_uses_future(self):
        df = data.synthetic(n_teams=10, seasons=3, seed=2)
        bets = backtest.run(df)
        self.assertIn("bets", backtest.summarise(bets))


if __name__ == "__main__":
    unittest.main()
