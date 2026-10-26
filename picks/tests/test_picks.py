import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from picks import backtest, cli, data, evaluate
from picks.xg import shot_xg
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


class XgTests(unittest.TestCase):
    def test_no_shot_columns_means_no_proxy(self):
        df = data.synthetic(n_teams=10, seasons=2, seed=1)
        self.assertIsNone(shot_xg(df.drop(columns=["HS", "AS", "HST", "AST"])))

    def test_coefficients_non_negative_and_on_target_worth_more(self):
        df = data.synthetic(n_teams=12, seasons=3, seed=1)
        _, _, coef = shot_xg(df)
        self.assertTrue((coef >= 0).all())

    def test_xg_weight_changes_ratings_but_still_normalised(self):
        df = data.synthetic(n_teams=12, seasons=3, seed=1)
        base = PoissonModel(xg_weight=0.0).fit(df)
        blend = PoissonModel(xg_weight=0.7).fit(df)
        self.assertNotEqual(base.attack["T00"], blend.attack["T00"])
        pr = blend.predict("T00", "T01")
        self.assertAlmostEqual(pr["H"] + pr["D"] + pr["A"], 1.0, places=6)

    def test_missing_shots_fall_back_to_goals(self):
        df = data.synthetic(n_teams=12, seasons=3, seed=1)
        df.loc[df.index[:50], ["HS", "AS", "HST", "AST"]] = float("nan")
        PoissonModel(xg_weight=1.0).fit(df)  # must not raise or produce NaN ratings
        self.assertFalse(np.isnan(list(PoissonModel(xg_weight=1.0).fit(df).attack.values())).any())


class EvaluateTests(unittest.TestCase):
    def test_scoring_rules(self):
        sure = {"H": 1.0, "D": 0.0, "A": 0.0}
        self.assertEqual(evaluate.rps(sure, "H"), 0.0)
        self.assertAlmostEqual(evaluate.rps(sure, "A"), 1.0)
        self.assertAlmostEqual(evaluate.log_loss({"H": 0.5, "D": 0.3, "A": 0.2}, "H"), 0.6931, places=3)

    def test_table_has_models_and_both_market_benchmarks(self):
        df = data.synthetic(n_teams=10, seasons=3, seed=2)
        table = evaluate.evaluate(df, xg_weights=(0.0, 0.5))
        self.assertEqual(len(table), 4)
        self.assertTrue((table["matches"] > 0).all())


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
