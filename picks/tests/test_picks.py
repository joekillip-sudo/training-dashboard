import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from picks import backtest, calibration, cli, data, evaluate, ledger, understat
from picks.xg import shot_xg
from picks.model import PoissonModel
from picks.value import devig, find_picks, kelly_fraction


class DataTests(unittest.TestCase):
    def test_season_codes_roll_over_in_july(self):
        self.assertEqual(data.season_codes(2, date(2026, 10, 26))[-1], "2627")
        self.assertEqual(data.season_codes(2, date(2027, 3, 1))[-1], "2627")
        self.assertEqual(data.season_codes(3, date(2026, 6, 30)), ["2324", "2425", "2526"])


class PipelineTests(unittest.TestCase):
    def _run(self, tmp, extra=()):
        argv = ["picks", "picks", "--data", str(tmp / "data"), "--fixtures", str(tmp / "fx.csv"),
                "--out", str(tmp / "ledger.json"), *extra]
        with mock.patch.object(sys, "argv", argv):
            cli.main()
        return json.loads((tmp / "ledger.json").read_text())

    def _setup(self, tmp):
        hist = data.synthetic(n_teams=12, seasons=3, noise=0.25, seed=4)
        (tmp / "data").mkdir()
        out = hist.copy()
        out["Date"] = out["Date"].dt.strftime("%d/%m/%Y")
        out.to_csv(tmp / "data" / "SYN_2526.csv", index=False)
        soon = (date.today() + timedelta(days=1)).strftime("%d/%m/%Y")
        far = (date.today() + timedelta(days=20)).strftime("%d/%m/%Y")
        pd.DataFrame([
            {"Div": "SYN", "Date": soon, "HomeTeam": "T00", "AwayTeam": "T01", "B365H": 9.0, "B365D": 5.0, "B365A": 1.3},
            {"Div": "SYN", "Date": soon, "HomeTeam": "T02", "AwayTeam": "T03", "B365H": None, "B365D": None, "B365A": None},
            {"Div": "SYN", "Date": far, "HomeTeam": "T04", "AwayTeam": "T05", "B365H": 9.0, "B365D": 5.0, "B365A": 1.3},
            {"Div": "XXX", "Date": soon, "HomeTeam": "A", "AwayTeam": "B", "B365H": 2.0, "B365D": 3.0, "B365A": 4.0},
        ]).to_csv(tmp / "fx.csv", index=False)

    def test_ledger_end_to_end_is_idempotent_and_hindsight_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._setup(tmp)
            first = self._run(tmp, ["--seed-backtest"])
            live = [b for b in first["bets"] if b["source"] == "live"]
            back = [b for b in first["bets"] if b["source"] == "backtest"]
            self.assertTrue(first["live_from"])
            self.assertTrue(back and all(b["result"] in "HDA" for b in back))
            for b in live:  # unknown league, odds-less and far-future fixtures are skipped
                self.assertEqual((b["league"], b["home"], b["away"]), ("SYN", "T00", "T01"))
                self.assertIsNone(b["result"])
            frozen = {b["id"]: dict(b) for b in live}
            second = self._run(tmp, ["--seed-backtest"])  # rerun changes nothing
            self.assertEqual(len(second["bets"]), len(first["bets"]))
            for b in second["bets"]:
                if b["id"] in frozen:
                    self.assertEqual(b, frozen[b["id"]])


class LedgerTests(unittest.TestCase):
    def test_settle_matches_result_and_records_closing_price(self):
        book = {"floor": ledger.FLOOR, "live_from": "2025-08-01", "bets": [
            {"id": "x", "league": "E0", "date": "2025-08-16", "home": "Arsenal", "away": "Chelsea", "outcome": "H",
             "odds": 2.5, "model_p": 0.5, "market_p": 0.4, "source": "live", "result": None, "close_p": None},
            {"id": "y", "league": "E0", "date": "2025-08-16", "home": "Spurs", "away": "Everton", "outcome": "A",
             "odds": 3.0, "model_p": 0.4, "market_p": 0.3, "source": "live", "result": None, "close_p": None}]}
        df = pd.DataFrame([{"League": "E0", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "Date": pd.Timestamp("2025-08-17"),
                            "FTR": "H", "PSCH": 2.0, "PSCD": 3.5, "PSCA": 4.0}])
        self.assertEqual(ledger.settle(book, df), 1)
        self.assertEqual(book["bets"][0]["result"], "H")
        self.assertGreater(book["bets"][0]["close_p"], 0.4)
        self.assertIsNone(book["bets"][1]["result"])  # no result yet stays open

    def test_save_load_roundtrip(self):
        book = {"floor": ledger.FLOOR, "live_from": None, "generated": None,
                "bets": [{"id": "b", "date": "2025-01-02"}, {"id": "a", "date": "2025-01-01"}]}
        with tempfile.TemporaryDirectory() as tmp:
            ledger.save(book, Path(tmp, "l.json"))
            back = ledger.load(Path(tmp, "l.json"))
        self.assertEqual([b["id"] for b in back["bets"]], ["a", "b"])


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
        table = evaluate.evaluate(df, variants=[(0.0, "shots"), (0.5, "shots")])
        self.assertEqual(len(table), 4)
        self.assertTrue((table["matches"] > 0).all())


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.df = data.synthetic(n_teams=12, seasons=3, noise=0.25, seed=6)
        self.fc = calibration.collect(self.df)

    def test_reliability_counts_every_forecast_once(self):
        t = calibration.reliability(self.fc, "model")
        self.assertEqual(int(t["n"].sum()), 3 * len(self.fc))
        self.assertTrue(((t["observed"] >= 0) & (t["observed"] <= 1)).all())

    def test_perfectly_calibrated_forecast_has_zero_gap(self):
        fc = pd.DataFrame({"ftr": ["H"] * 5 + ["A"] * 5, **{f"model_{o}": [0.5 if o == "H" else 0.0 if o == "D" else 0.5] * 10 for o in "HDA"}})
        t = calibration.reliability(fc, "model")
        self.assertAlmostEqual(calibration.ece(t), 0.0)

    def test_blend_endpoints_match_pure_sources(self):
        s = calibration.blend_log_loss(self.fc, "close")
        self.assertEqual(s.index[0], 0.0)
        self.assertEqual(s.index[-1], 1.0)
        self.assertGreater(s.iloc[0], 0)

    def test_pick_calibration_bands(self):
        bets = backtest.run(self.df, min_ev=0.02, min_edge=0.01)
        t = calibration.pick_calibration(bets)
        self.assertEqual(int(t["bets"].sum()), len(bets))


def _page(matches):
    """Build HTML the way Understat embeds data: JSON in a JS string with \\xHH escapes."""
    raw = json.dumps(matches)
    esc = "".join(c if c.isalnum() else "\\x%02X" % ord(c) for c in raw)
    return f"<html><script>var datesData = JSON.parse('{esc}');</script></html>"


def _game(h, a, hx, ax, when, played=True):
    return {"isResult": played, "datetime": when, "h": {"title": h}, "a": {"title": a}, "xG": {"h": str(hx), "a": str(ax)}}


class UnderstatTests(unittest.TestCase):
    def test_parse_matches_skips_unplayed_and_decodes_escapes(self):
        html = _page([
            _game("Manchester United", "Nott'm Forest", 1.8, 0.4, "2025-08-17 15:30:00"),
            _game("Arsenal", "Chelsea", 0, 0, "2025-12-01 20:00:00", played=False),
        ])
        df = understat.parse_matches(html)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "AwayTeam"], "Nott'm Forest")
        self.assertAlmostEqual(df.loc[0, "HxG"], 1.8)

    def test_parse_fails_loudly_when_layout_changes(self):
        with self.assertRaises(ValueError):
            understat.parse_matches("<html>nothing here</html>")

    def test_map_teams_alias_exact_fuzzy_and_unmatched(self):
        fd = {"Man United", "Arsenal", "Leverkusen", "Brentford"}
        mapping, unmatched = understat.map_teams(
            {"Manchester United", "Arsenal FC", "Bayer Leverkusen", "Totally Unknown"}, fd)
        self.assertEqual(mapping["Manchester United"], "Man United")
        self.assertEqual(mapping["Arsenal FC"], "Arsenal")
        self.assertEqual(mapping["Bayer Leverkusen"], "Leverkusen")
        self.assertEqual(unmatched, ["Totally Unknown"])

    def test_map_teams_rejects_ambiguous_collisions(self):
        mapping, unmatched = understat.map_teams({"Real Betis", "Betis"}, {"Betis"})
        self.assertEqual(mapping, {})
        self.assertEqual(unmatched, ["Betis", "Real Betis"])

    def test_attach_xg_matches_within_date_tolerance_only(self):
        df = pd.DataFrame({
            "League": ["E0", "E0"], "HomeTeam": ["Man United", "Man United"], "AwayTeam": ["Arsenal", "Arsenal"],
            "Date": pd.to_datetime(["2025-08-17", "2026-02-01"]), "FTHG": [1, 2], "FTAG": [0, 2]})
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "E0_2526.csv").write_text(
                "Date,HomeTeam,AwayTeam,HxG,AxG\n2025-08-18,Manchester United,Arsenal,1.5,0.7\n")
            out = understat.attach_xg(df, tmp, Path(tmp, "none.json"))
        self.assertAlmostEqual(out.loc[0, "HxG"], 1.5)
        self.assertTrue(np.isnan(out.loc[1, "HxG"]))  # the other fixture is months away

    def test_fetch_writes_csv_and_survives_failures(self):
        page = _page([_game("Arsenal", "Chelsea", 2.0, 1.0, "2025-08-20 19:00:00")]).encode()
        opener = lambda req: mock.Mock(read=lambda: page)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(understat.fetch(["E0"], ["2526"], tmp, delay=0, opener=opener), 1)
            self.assertTrue(Path(tmp, "E0_2526.csv").exists())
            def boom(req):
                raise OSError("blocked")
            self.assertEqual(understat.fetch(["E0"], ["2526"], tmp, delay=0, opener=boom), 0)

    def test_load_attaches_sibling_xg_dir_and_model_uses_it(self):
        hist = data.synthetic(n_teams=12, seasons=3, noise=0.25, seed=4)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "data").mkdir()
            Path(tmp, "xg_data").mkdir()
            fd = hist.copy()
            fd["Date"] = fd["Date"].dt.strftime("%d/%m/%Y")
            fd.to_csv(Path(tmp, "data", "SYN_2526.csv"), index=False)
            real = hist[["Date", "HomeTeam", "AwayTeam"]].assign(HxG=hist["FTHG"] * 0.5 + 0.4, AxG=hist["FTAG"] * 0.5 + 0.4)
            real.to_csv(Path(tmp, "xg_data", "SYN_2526.csv"), index=False)
            df = data.load(Path(tmp, "data"))
        self.assertGreater(df["HxG"].notna().mean(), 0.95)
        auto = PoissonModel(xg_weight=0.8, xg_source="auto").fit(df)
        shots = PoissonModel(xg_weight=0.8, xg_source="shots").fit(df)
        self.assertNotAlmostEqual(auto.attack["T00"], shots.attack["T00"], places=4)
        self.assertEqual(PoissonModel(xg_weight=0.8, xg_source="shots").fit(df).attack["T00"], shots.attack["T00"])


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
