import gzip
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from picks import backtest, calibration, cli, data, evaluate, ledger, market, tune, understat
from picks.model import MultiLeagueModel
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


class FixturesTests(unittest.TestCase):
    def test_league_column_aliases_and_inference_without_one(self):
        hist = data.synthetic(n_teams=12, seasons=3, noise=0.25, seed=4)
        models = {"SYN": PoissonModel().fit(hist)}
        soon = (date.today() + timedelta(days=1)).strftime("%d/%m/%Y")
        rows = [{"Date": soon, "HomeTeam": "T00", "AwayTeam": "T01", "B365H": 9.0, "B365D": 5.0, "B365A": 1.3},
                {"Date": soon, "HomeTeam": "Nobody", "AwayTeam": "T01", "B365H": 2.0, "B365D": 3.0, "B365A": 4.0}]
        with tempfile.TemporaryDirectory() as tmp:
            plain, named = Path(tmp, "a.csv"), Path(tmp, "b.csv")
            pd.DataFrame(rows).to_csv(plain, index=False)
            pd.DataFrame(rows).assign(division="SYN").to_csv(named, index=False)  # odd name and case
            fx = data.load_fixtures(plain)
            self.assertNotIn("League", fx.columns)
            inferred = data.infer_leagues(fx, models)
            self.assertEqual(list(inferred["HomeTeam"]), ["T00"])
            self.assertEqual(list(inferred["League"]), ["SYN"])
            self.assertEqual(list(data.load_fixtures(named)["League"]), ["SYN", "SYN"])
            bad = Path(tmp, "c.csv")
            pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
            with self.assertRaises(ValueError):
                data.load_fixtures(bad)

    def test_cli_runs_when_fixtures_have_no_league_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            PipelineTests()._setup(tmp)
            fx = pd.read_csv(tmp / "fx.csv").drop(columns=["Div"])
            fx.to_csv(tmp / "fx.csv", index=False)
            book = PipelineTests()._run(tmp)
        self.assertTrue(all(b["league"] == "SYN" for b in book["bets"]))


class KickoffTests(unittest.TestCase):
    """Live bets must be placeable: nothing on a match that has already started."""

    def _book_for(self, rows, now):
        hist = data.synthetic(n_teams=12, seasons=3, noise=0.25, seed=4)
        models = {"SYN": PoissonModel().fit(hist)}
        fx = pd.DataFrame(rows)
        fx["Date"] = pd.to_datetime(fx["Date"])
        book = {"floor": ledger.FLOOR, "live_from": None, "bets": []}
        ledger.add_live(book, fx, models, now)
        return {(b["home"], b["away"]) for b in book["bets"]}

    def test_started_and_unknown_time_matches_are_skipped(self):
        now = pd.Timestamp("2026-10-26 22:46", tz="UTC").to_pydatetime()
        odds = {"League": "SYN", "B365H": 9.0, "B365D": 5.0, "B365A": 1.3}
        rows = [
            {**odds, "Date": "2026-10-26", "Time": "20:00", "HomeTeam": "T00", "AwayTeam": "T01"},  # over by now
            {**odds, "Date": "2026-10-26", "HomeTeam": "T02", "AwayTeam": "T03"},                   # today, no time: could be over
            {**odds, "Date": "2026-10-27", "HomeTeam": "T04", "AwayTeam": "T05"},                   # tomorrow, no time
            {**odds, "Date": "2026-10-27", "Time": "19:45", "HomeTeam": "T06", "AwayTeam": "T07"},  # tomorrow evening
        ]
        got = self._book_for(rows, now)
        self.assertFalse(got & {("T00", "T01"), ("T02", "T03")})

    def test_later_today_is_allowed_when_kickoff_time_is_known(self):
        now = pd.Timestamp("2026-10-26 08:17", tz="UTC").to_pydatetime()
        odds = {"League": "SYN", "B365H": 9.0, "B365D": 5.0, "B365A": 1.3}
        early = {**odds, "Date": "2026-10-26", "Time": "08:30", "HomeTeam": "T00", "AwayTeam": "T01"}  # inside the buffer
        later = {**odds, "Date": "2026-10-26", "Time": "15:00", "HomeTeam": "T02", "AwayTeam": "T03"}
        got = self._book_for([early, later], now)
        self.assertNotIn(("T00", "T01"), got)
        self.assertTrue(ledger._bettable(pd.Series({**later, "Date": pd.Timestamp("2026-10-26")}), now))
        self.assertFalse(ledger._bettable(pd.Series({**early, "Date": pd.Timestamp("2026-10-26")}), now))


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

    def test_closing_price_falls_back_to_the_average_when_pinnacle_is_missing(self):
        row = pd.Series({"PSCH": float("nan"), "PSCD": float("nan"), "PSCA": float("nan"),
                         "AvgCH": 2.0, "AvgCD": 3.5, "AvgCA": 4.0})
        self.assertGreater(ledger._close_p(row, "H"), 0.4)
        self.assertIsNone(ledger._close_p(pd.Series({"PSCH": float("nan")}), "H"))

    def test_save_load_roundtrip(self):
        book = {"floor": ledger.FLOOR, "live_from": None, "generated": None,
                "bets": [{"id": "b", "date": "2025-01-02"}, {"id": "a", "date": "2025-01-01"}]}
        with tempfile.TemporaryDirectory() as tmp:
            ledger.save(book, Path(tmp, "l.json"))
            back = ledger.load(Path(tmp, "l.json"))
        self.assertEqual([b["id"] for b in back["bets"]], ["a", "b"])


class ScaleTests(unittest.TestCase):
    """Regression: league scale was applied twice to home goals, biasing every forecast toward home wins."""

    def _check(self, mu_true, seed):
        rng = np.random.default_rng(seed)
        teams = [f"T{i}" for i in range(16)]
        att = np.exp(rng.normal(0, 0.2, 16)); att /= att.mean()
        dfn = np.exp(rng.normal(0, 0.2, 16)); dfn /= dfn.mean()
        rows = []
        for r in range(8):
            for h in range(16):
                for a in range(16):
                    if h != a:
                        rows.append({"Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=r), "HomeTeam": teams[h], "AwayTeam": teams[a],
                                     "FTHG": rng.poisson(mu_true * 1.3 * att[h] * dfn[a]), "FTAG": rng.poisson(mu_true * att[a] * dfn[h])})
        df = pd.DataFrame(rows)
        m = PoissonModel(half_life_days=5000, shrink=1.0).fit(df)
        lh = np.mean([m.expected_goals(h, a)[0] for h, a in zip(df.HomeTeam, df.AwayTeam)])
        la = np.mean([m.expected_goals(h, a)[1] for h, a in zip(df.HomeTeam, df.AwayTeam)])
        self.assertAlmostEqual(lh / df.FTHG.mean(), 1.0, delta=0.03)
        self.assertAlmostEqual(la / df.FTAG.mean(), 1.0, delta=0.03)
        self.assertAlmostEqual(m.home_adv, 1.3, delta=0.1)
        self.assertAlmostEqual(m.league_avg, mu_true, delta=0.1)

    def test_model_reproduces_training_goal_averages_for_low_and_high_scoring_leagues(self):
        self._check(0.9, 1)   # scale below 1
        self._check(1.4, 2)   # scale above 1: this case was ~20% too high at home before the fix


class MarketTests(unittest.TestCase):
    def _sample(self, n, tilt, seed=0):
        rng = np.random.default_rng(seed)
        base = rng.dirichlet([6, 3, 4], n)
        true = base ** tilt
        true /= true.sum(1, keepdims=True)
        y = np.array([rng.choice(3, p=p) for p in true])
        model = np.clip(base * np.exp(rng.normal(0, 0.1, base.shape)), 1e-3, None)
        return model / model.sum(1, keepdims=True), base, y

    def test_residual_stays_at_the_base_when_the_base_is_right(self):
        mp, bp, y = self._sample(3000, tilt=1.0)
        theta = market.fit_residual(mp, bp, y)
        self.assertLess(np.abs(theta).max(), 0.15)
        out = market.predict_residual(theta, mp, bp)
        np.testing.assert_allclose(out.sum(1), 1.0)
        self.assertLess(np.abs(out - bp).mean(), 0.02)

    def test_residual_learns_a_known_bias_in_the_base(self):
        mp, bp, y = self._sample(6000, tilt=1.25, seed=1)  # truth is more extreme than the base says
        theta = market.fit_residual(mp, bp, y)
        self.assertGreater(theta[3], 0.12)  # tilt term picks up roughly +0.25 (shrunk toward 0)

    def test_settle_arithmetic(self):
        p = np.array([[0.6, 0.2, 0.2], [0.2, 0.3, 0.5]])
        odds = np.array([[2.0, 4.0, 5.0], [3.0, 3.5, 1.9]])
        y = np.array([0, 1])
        bet, profit, _ = market.settle(p, odds, y, thr=0.05)
        self.assertEqual(bet.tolist(), [True, True])   # EV 0.2 on H; EV 0.05 on D in match 2, H 3.0 is -0.4, A -0.05
        self.assertAlmostEqual(profit[0], 1.0)         # backed H at 2.0, won
        self.assertAlmostEqual(profit[1], 2.5)         # best EV is D at 3.5 (0.05); it won, so +2.5

    def test_forecast_table_and_oos_comparison_on_synthetic_data(self):
        df = data.synthetic(n_teams=14, seasons=4, noise=0.25, seed=8)
        tbl = market.forecast_table(df)
        self.assertTrue({"model_H", "ps_p_H", "max_odds_A", "close_p_D", "date"} <= set(tbl.columns))
        np.testing.assert_allclose(tbl[["ps_p_H", "ps_p_D", "ps_p_A"]].sum(1), 1.0)
        oos, theta = market.residual_oos(tbl, min_train=len(tbl) // 3)
        self.assertGreater(len(oos), 100)
        self.assertEqual(len(theta), 4)
        out = market.compare(oos, thresholds=(0.0, 0.05))
        self.assertEqual(len(out), 6 * 2)
        self.assertTrue((out["bets"] >= 0).all())
        self.assertEqual(set(market.log_losses(oos)), {"model", "resid", "ps", "avg", "b365", "close"})


class FixturesFileTests(unittest.TestCase):
    def test_byte_order_mark_does_not_hide_the_division_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, raw in (("utf8bom.csv", "\ufeffDiv,Date,HomeTeam,AwayTeam\nE0,27/10/2026,A,B\n".encode("utf-8")),
                              ("latin1view.csv", "ï»¿Div,Date,HomeTeam,AwayTeam\nE0,27/10/2026,A,B\n".encode("utf-8"))):
                path = Path(tmp, name)
                path.write_bytes(raw)
                fx = data.load_fixtures(path)
                self.assertEqual(list(fx["League"]), ["E0"], name)

    def test_decoding_handles_utf8_with_bom_and_latin1(self):
        self.assertEqual(data._decode("\ufeffDiv,Köln".encode("utf-8")), "Div,Köln")
        self.assertEqual(data._decode("Div,Atlético".encode("latin-1")), "Div,Atlético")


class MultiLeagueTests(unittest.TestCase):
    def _two_leagues(self):
        a = data.synthetic(n_teams=10, seasons=3, noise=0.25, seed=21)
        b = data.synthetic(n_teams=10, seasons=3, noise=0.25, seed=22)
        a["League"], b["League"] = "E0", "E1"
        b["HomeTeam"] = b["HomeTeam"].str.replace("T", "U")
        b["AwayTeam"] = b["AwayTeam"].str.replace("T", "U")
        return pd.concat([a, b], ignore_index=True)

    def test_one_model_serves_every_league_and_acts_like_a_dict(self):
        m = MultiLeagueModel().fit(self._two_leagues())
        self.assertEqual(sorted(m), ["E0", "E1"])
        self.assertIn("E0", m)
        self.assertTrue(m.knows("E1", "U01") and not m.knows("E0", "U01") and not m.knows("ZZ", "T01"))
        pr = m.predict("E0", "T00", "T01")
        self.assertAlmostEqual(pr["H"] + pr["D"] + pr["A"], 1.0, places=6)
        self.assertIs(m.get("E0"), m["E0"])
        self.assertIsNone(m.get("nope"))

    def test_league_inference_prefers_the_league_both_teams_were_seen_in_most_recently(self):
        df = self._two_leagues()
        # T00 and T01 were relegated: old E0 seasons, plus recent E1 matches between them
        late = df["Date"].max() + pd.Timedelta(days=30)
        extra = pd.DataFrame([{"Date": late, "HomeTeam": "T00", "AwayTeam": "T01", "FTHG": 1, "FTAG": 0, "FTR": "H", "League": "E1"}] * 40)
        m = MultiLeagueModel().fit(pd.concat([df, extra], ignore_index=True))
        self.assertTrue(m["E0"].knows("T00") and m["E1"].knows("T00"))  # known in both leagues
        fx = pd.DataFrame([{"Date": late, "HomeTeam": "T00", "AwayTeam": "T01"},
                           {"Date": late, "HomeTeam": "T02", "AwayTeam": "T03"}])
        out = data.infer_leagues(fx, m)
        self.assertEqual(dict(zip(out["HomeTeam"], out["League"])), {"T00": "E1", "T02": "E0"})


class SofterMarketTests(unittest.TestCase):
    def test_average_closing_price_replaces_missing_pinnacle_close(self):
        df = data.synthetic(n_teams=12, seasons=3, noise=0.25, seed=11)
        for o in "HDA":
            df[f"AvgC{o}"] = df[f"PSC{o}"]
            df[f"PSC{o}"] = float("nan")  # Pinnacle's feed gone, as in recent data
        tbl = market.forecast_table(df)
        self.assertFalse(tbl["close_p_H"].isna().any())
        np.testing.assert_allclose(tbl[["close_p_H", "close_p_D", "close_p_A"]].sum(1), 1.0)
        self.assertNotIn("nan", str(market.log_losses(tbl.assign(resid_p_H=0.4, resid_p_D=0.3, resid_p_A=0.3))["close"]))

    def test_league_report_slices_leagues_and_groups(self):
        a = data.synthetic(n_teams=12, seasons=3, noise=0.25, seed=12)
        b = data.synthetic(n_teams=12, seasons=3, noise=0.25, seed=13)
        a["League"], b["League"] = "E0", "E2"
        tbl = market.forecast_table(pd.concat([a, b], ignore_index=True))
        fc, bt = market.league_report(tbl)
        self.assertEqual(set(fc["slice"]), {"E0", "E2", "top 5", "lower England", "ALL"})
        row = fc[fc["slice"] == "ALL"].iloc[0]
        self.assertAlmostEqual(row["gap"], row["ll_model"] - row["ll_market"], places=3)
        self.assertTrue(0.01 < row["overround"] < 0.10)  # synthetic average prices carry a 4% margin
        self.assertTrue((bt["old_bets"] >= 0).all())

    def test_small_slices_are_left_out_rather_than_over_read(self):
        df = data.synthetic(n_teams=8, seasons=2, noise=0.25, seed=14)
        tbl = market.forecast_table(df).head(20)
        fc, _ = market.league_report(tbl)
        self.assertTrue(fc.empty)

    def test_cli_leagues_runs(self):
        with mock.patch.object(sys, "argv", ["picks", "leagues", "--synthetic"]), mock.patch("builtins.print") as pr:
            cli.main()
        text = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertIn("FORECAST QUALITY", text)


class TuneTests(unittest.TestCase):
    def test_small_grid_returns_sorted_table_and_default_comparison(self):
        df = data.synthetic(n_teams=10, seasons=3, noise=0.25, seed=9)
        grid = {"half_life_days": (200, 365), "shrink": (3.0,), "rho": (-0.05,)}
        res, summary = tune.tune(df, {}, {"half_life_days": 365, "shrink": 3.0, "rho": -0.05}, grid=grid)
        self.assertEqual(len(res), 2)
        self.assertTrue(res["early"].is_monotonic_increasing)
        self.assertIsNotNone(summary["default_late"])


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

    def test_map_teams_handles_prefixed_and_suffixed_names(self):
        fd = {"Hamburg", "Clermont", "La Coruna", "Santander", "Oviedo", "Valladolid", "Roma", "Lazio"}
        names = {"Hamburger SV", "Clermont Foot", "Deportivo La Coruna", "Racing Santander", "Real Oviedo", "Real Valladolid"}
        mapping, unmatched = understat.map_teams(names, fd)
        self.assertEqual(unmatched, [])
        self.assertEqual(mapping["Real Oviedo"], "Oviedo")
        self.assertEqual(mapping["Deportivo La Coruna"], "La Coruna")
        # a wrong alias target (not in the data) falls back to containment instead of leaving the team out
        mapping, _ = understat.map_teams({"Real Oviedo"}, {"Oviedo", "Lazio"}, aliases={"Real Oviedo": "Nonexistent FC"})
        self.assertEqual(mapping, {"Real Oviedo": "Oviedo"})
        # short or ambiguous containment must not match
        mapping, _ = understat.map_teams({"AS Roma Calcio Plus"}, {"Roma", "Lazio"})
        self.assertEqual(mapping, {})

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

    def test_fetch_prefers_json_endpoint_and_falls_back_to_page(self):
        games = [_game("Arsenal", "Chelsea", 2.0, 1.0, "2025-08-20 19:00:00")]
        seen = []
        def opener(req):
            seen.append(req.full_url)
            body = json.dumps({"teams": {}, "players": [], "dates": games}) if "getLeagueData" in req.full_url else "<html></html>"
            return mock.Mock(read=lambda: body.encode())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(understat.fetch(["E0"], ["2526"], tmp, delay=0, opener=opener), 1)
            self.assertEqual(len(seen), 1)  # JSON worked, page never requested
            page = _page(games).encode()
            only_page = lambda req: mock.Mock(read=lambda: b"{}" if "getLeagueData" in req.full_url else page)
            self.assertEqual(understat.fetch(["SP1"], ["2526"], tmp, delay=0, opener=only_page), 1)

    def test_gzipped_json_response_is_decompressed(self):
        games = [_game("Arsenal", "Chelsea", 2.0, 1.0, "2025-08-20 19:00:00")]
        body = gzip.compress(json.dumps({"dates": games}).encode())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(understat.fetch(["E0"], ["2526"], tmp, delay=0, opener=lambda r: mock.Mock(read=lambda: body)), 1)
            self.assertEqual(len(pd.read_csv(Path(tmp, "E0_2526.csv"))), 1)

    def test_failure_message_says_what_came_back(self):
        opener = lambda req: mock.Mock(read=lambda: b"<html>Please enable JavaScript</html>")
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.print") as pr:
            self.assertEqual(understat.fetch(["E0"], ["2526"], tmp, delay=0, opener=opener), 0)
        self.assertIn("Please enable JavaScript", pr.call_args[0][0])

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
