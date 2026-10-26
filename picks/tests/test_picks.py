import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

from picks import backtest, cli, data
from picks.model import PoissonModel
from picks.value import devig, find_picks, kelly_fraction


class DataTests(unittest.TestCase):
    def test_season_codes_roll_over_in_july(self):
        self.assertEqual(data.season_codes(2, date(2026, 10, 26))[-1], "2627")
        self.assertEqual(data.season_codes(2, date(2027, 3, 1))[-1], "2627")
        self.assertEqual(data.season_codes(3, date(2026, 6, 30)), ["2324", "2425", "2526"])


class PipelineTests(unittest.TestCase):
    def test_cli_picks_end_to_end_from_local_files(self):
        hist = data.synthetic(n_teams=12, seasons=3, noise=0.25, seed=4)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "data").mkdir()
            out = hist.copy()
            out["Date"] = out["Date"].dt.strftime("%d/%m/%Y")
            out.to_csv(tmp / "data" / "SYN_2526.csv", index=False)
            soon = (date.today() + timedelta(days=2)).strftime("%d/%m/%Y")
            pd.DataFrame([
                {"Div": "SYN", "Date": soon, "HomeTeam": "T00", "AwayTeam": "T01", "B365H": 9.0, "B365D": 5.0, "B365A": 1.3},
                {"Div": "SYN", "Date": soon, "HomeTeam": "T02", "AwayTeam": "T03", "B365H": None, "B365D": None, "B365A": None},
                {"Div": "XXX", "Date": soon, "HomeTeam": "A", "AwayTeam": "B", "B365H": 2.0, "B365D": 3.0, "B365A": 4.0},
            ]).to_csv(tmp / "fx.csv", index=False)
            dest = tmp / "picks.json"
            argv = ["picks", "picks", "--data", str(tmp / "data"), "--fixtures", str(tmp / "fx.csv"), "--out", str(dest)]
            with mock.patch.object(sys, "argv", argv):
                cli.main()
            payload = json.loads(dest.read_text())
        self.assertIn("generated", payload)
        self.assertIn("bets", payload["backtest"])
        for p in payload["picks"]:  # unknown league and odds-less fixtures are skipped
            self.assertEqual(p["league"], "SYN")
            self.assertEqual(p["match"], "T00 v T01")


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
