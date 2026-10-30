import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tennis import data, fit, market, pipeline, report, synth
from tennis.elo import Elo, run_elo


def raw_match(winner, loser, day="2015-03-01", rnd="1st Round", **kw):
    r = {"Tournament": "Test Open", "Date": pd.Timestamp(day), "Series": "ATP250", "Surface": "Hard", "Round": rnd, "Best of": 3,
         "Winner": winner, "Loser": loser, "WRank": 10, "LRank": 50, "WPts": 1000, "LPts": 500, "Wsets": 2, "Lsets": 0,
         "Comment": "Completed", "PSW": 1.5, "PSL": 2.6, "AvgW": 1.5, "AvgL": 2.5, "MaxW": 1.55, "MaxL": 2.7, "B365W": 1.5, "B365L": 2.5}
    r.update(kw)
    return r


class OrientTests(unittest.TestCase):
    def test_listing_order_carries_no_information(self):
        """Who is listed as 'Winner' must not predict which player becomes player 1."""
        raw = synth.synth_raw(n_matches=6000, seed=2, retire_rate=0)
        o = data.orient(data.clean(raw, "ATP")[0])
        n = len(o)
        self.assertLess(abs(o["y"].mean() - 0.5), 4 * 0.5 / np.sqrt(n))     # player 1 wins half the time
        # a naive model that learns 'player 1 wins' from the first half cannot do better than chance on the second
        a, b = o["y"].iloc[: n // 2].mean(), o["y"].iloc[n // 2:].mean()
        self.assertLess(abs(a - b), 4 * 0.5 / np.sqrt(n / 2))
        # prices pointed at player 1 are not systematically shorter (player 1 is not the 'winner')
        self.assertLess(abs(np.nanmean(market.devig(o["ps_o1"], o["ps_o2"])) - 0.5), 0.02)

    def test_same_pair_same_player_one_regardless_of_who_won(self):
        """Orientation depends on the unordered pair, so swapping the winner/loser roles keeps the same player 1."""
        ab = data.orient(data.clean(pd.DataFrame([raw_match("Alpha A.", "Beta B.")]), "ATP")[0])
        ba = data.orient(data.clean(pd.DataFrame([raw_match("Beta B.", "Alpha A.")]), "ATP")[0])
        self.assertEqual(ab["p1"].iloc[0], ba["p1"].iloc[0])
        self.assertEqual(ab["y"].iloc[0] + ba["y"].iloc[0], 1)

    def test_odds_follow_the_players(self):
        o = data.orient(data.clean(pd.DataFrame([raw_match("Alpha A.", "Beta B.")]), "ATP")[0]).iloc[0]
        winner_is_p1 = o["y"] == 1
        self.assertEqual(o["ps_o1"], 1.5 if winner_is_p1 else 2.6)
        self.assertEqual(o["ps_o2"], 2.6 if winner_is_p1 else 1.5)
        self.assertEqual(o["r1"], 10 if winner_is_p1 else 50)


class CleanTests(unittest.TestCase):
    def test_retirements_walkovers_and_short_sets_removed(self):
        rows = [raw_match("A A.", "B B."), raw_match("C C.", "D D.", Comment="Retired"), raw_match("E E.", "F F.", Comment="Walkover"),
                raw_match("G G.", "H H.", Wsets=1, Lsets=0), raw_match("I I.", "J J.", Comment="Disqualified")]
        m, diag = data.clean(pd.DataFrame(rows), "ATP")
        self.assertEqual(list(m["winner"]), ["A A."])
        self.assertEqual(diag["dropped_unfinished"], 4)

    def test_levels_and_surfaces(self):
        self.assertEqual([data.level_of(x) for x in ("Grand Slam", "Masters 1000", "Masters", "ATP500", "ATP250", "International Gold", "International", "Premier", "Premier Mandatory", "WTA1000", "Tour Championships")],
                         ["Grand Slam", "Masters 1000", "Masters 1000", "500", "250", "500", "250", "500", "Masters 1000", "Masters 1000", "Finals"])
        m, _ = data.clean(pd.DataFrame([raw_match("A A.", "B B.", Surface="Carpet")]), "ATP")
        self.assertEqual(m["surface"].iloc[0], "hard")

    def test_invalid_prices_become_missing(self):
        m, _ = data.clean(pd.DataFrame([raw_match("A A.", "B B.", PSW=1.0, MaxW="n/a")]), "ATP")
        self.assertTrue(np.isnan(m["ps_w"].iloc[0]) and np.isnan(m["max_w"].iloc[0]))
        self.assertFalse(np.isnan(m["avg_w"].iloc[0]))

    def test_wta_tier_column_and_xlsx_round_trip(self):
        df = pd.DataFrame([raw_match("A A.", "B B.")]).rename(columns={"Series": "Tier"})
        df["Tier"] = "Premier"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "wta_2015.xlsx"
            df.to_excel(p, index=False)
            m, _ = data.clean(data.read_workbook(p), "WTA")
        self.assertEqual(m["level"].iloc[0], "500")
        self.assertEqual(len(m), 1)

    def test_link_parsing(self):
        html = '<a href="2023/2023.xlsx">a</a><a href="2023w/2023.xlsx">b</a><a href="2012/2012.zip">z</a><a href="2012/2012.xls">x</a><a href="other.zip">q</a>'
        got = data.parse_links(html, "http://www.tennis-data.co.uk/alldata.php")
        self.assertEqual(got[("atp", 2023)], ["http://www.tennis-data.co.uk/2023/2023.xlsx"])
        self.assertEqual(got[("wta", 2023)], ["http://www.tennis-data.co.uk/2023w/2023.xlsx"])
        self.assertEqual(got[("atp", 2012)][0].endswith(".xls"), True)
        self.assertFalse(data.looks_like_workbook(b"<html>404</html>"))


class EloTests(unittest.TestCase):
    def _oriented(self, **kw):
        return data.orient(data.clean(synth.synth_raw(**kw), "ATP")[0])

    def test_no_look_ahead(self):
        """Changing a match's result, or deleting everything after it, cannot change its (or earlier) predictions."""
        o = self._oriented(n_matches=1500, n_players=30, seed=4, retire_rate=0)
        base, _ = run_elo(o)
        k = 900
        flipped = o.copy()
        flipped.loc[k, "y"] = 1 - flipped.loc[k, "y"]
        alt, _ = run_elo(flipped)
        same_group = (o["date"] == o.loc[k, "date"]) & (o["rround"] == o.loc[k, "rround"])
        before = (o["date"] < o.loc[k, "date"]) | ((o["date"] == o.loc[k, "date"]) & (o["rround"] <= o.loc[k, "rround"]))
        np.testing.assert_allclose(base.loc[before, "elo_diff"], alt.loc[before, "elo_diff"])
        self.assertTrue(same_group.sum() >= 1)
        trunc, _ = run_elo(o.iloc[: k + 1].copy())
        np.testing.assert_allclose(base["elo_diff"].iloc[:k], trunc["elo_diff"].iloc[:k])

    def test_walk_forward_fit_ignores_future(self):
        o = self._oriented(n_matches=5000, n_players=40, seed=5, retire_rate=0)
        f, _ = run_elo(o)
        X = np.column_stack([f["elo_diff"] * np.log(10) / 400])
        y = f["y"].to_numpy()
        cut = pd.Timestamp("2012-01-01")
        p1, _ = fit.walk_forward_fit(f["date"], X, y, test_start="2010-01-01", min_train=300)
        y2 = y.copy()
        y2[(f["date"] >= cut).to_numpy()] = 1 - y2[(f["date"] >= cut).to_numpy()]  # corrupt the future
        p2, _ = fit.walk_forward_fit(f["date"], X, y2, test_start="2010-01-01", min_train=300)
        early = (f["date"] < cut).to_numpy() & np.isfinite(p1)
        self.assertTrue(early.sum() > 100)
        np.testing.assert_allclose(p1[early], p2[early])

    def test_elo_is_calibrated_and_beats_coin_flip_on_a_known_world(self):
        o = self._oriented(n_matches=20000, n_players=80, seed=6, retire_rate=0, noise_sd=0)
        f, _ = run_elo(o)
        w = f[(f["n1"] >= 15) & (f["n2"] >= 15)]
        self.assertLess(market.log_loss(w["elo_p"], w["y"]).mean(), np.log(2) - 0.05)
        slope, se = fit.slope_with_se(w["elo_p"], w["y"])
        self.assertTrue(0.7 < slope < 1.15, slope)        # a scale applied twice would push this far from 1
        self.assertLess(abs(w["elo_p"].mean() - w["y"].mean()), 0.01)

    def test_single_update_is_one_scale(self):
        e = Elo()
        before = e.diff("a", "b", "hard")
        e.update("a", "b", "hard")
        self.assertEqual(before, 0)
        # equal ratings, expected 0.5: winner gains k*(0.5), loser loses k*(0.5), k = 250/5**0.4
        k = 250 / 5 ** 0.4
        self.assertAlmostEqual(e.ro["a"], 1500 + k * 0.5)
        self.assertAlmostEqual(e.ro["b"], 1500 - k * 0.5)
        self.assertAlmostEqual(Elo.expected(400), 10 / 11)

    def test_same_round_matches_do_not_inform_each_other(self):
        """Two matches in one group: the second may not see the first's result."""
        rows = [raw_match("A A.", "B B."), raw_match("A A.", "C C.")]   # same date and round, A in both
        o = data.orient(data.clean(pd.DataFrame(rows), "ATP")[0])
        f, _ = run_elo(o)
        self.assertTrue((f["n1"] + f["n2"] == 0).all())


class MarketTests(unittest.TestCase):
    def test_devig_and_overround(self):
        p = market.devig([1.9, 1.9], [1.9, 1.9])
        np.testing.assert_allclose(p, 0.5)
        self.assertAlmostEqual(float(market.overround(1.9, 1.9)), 2 / 1.9 - 1)
        self.assertAlmostEqual(float(market.devig(1.5, 3.0)), (1 / 1.5) / (1 / 1.5 + 1 / 3))

    def test_settle_flat_stakes(self):
        # model sees value on player 2 in match 1 (wins), on player 1 in match 2 (loses), none in match 3
        p1 = np.array([0.30, 0.60, 0.50])
        o1 = np.array([1.5, 2.0, 1.8])
        o2 = np.array([3.5, 1.9, 1.8])
        y = np.array([0, 0, 1])
        bet, profit, odds, _ = market.settle(p1, o1, o2, y, 0.05)
        self.assertEqual(list(bet), [True, True, False])
        np.testing.assert_allclose(profit[:2], [2.5, -1.0])
        r = market.summarise(bet, profit, odds, np.full(3, np.nan), np.array([False, True, True]))
        self.assertAlmostEqual(r["roi"], 0.75)
        self.assertEqual(r["bets"], 2)

    def test_moe_and_reliability(self):
        n, m, moe = market.mean_moe([1.0, -1.0] * 50)
        self.assertEqual(n, 100)
        self.assertAlmostEqual(moe, 1.96 * np.std([1.0, -1.0] * 50, ddof=1) / 10)
        rel = market.reliability(np.array([0.9, 0.1, 0.6, 0.4]), np.array([1, 0, 0, 0]))
        self.assertEqual(int(rel["n"].sum()), 4)   # each match counted once

    def test_fit_logit_recovers_coefficient(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40000, 1))
        y = (rng.random(40000) < fit.sigmoid(1.3 * x[:, 0])).astype(float)
        self.assertAlmostEqual(float(fit.fit_logit(x, y, lam=0.0)[0]), 1.3, delta=0.05)
        self.assertAlmostEqual(fit.slope_with_se(fit.sigmoid(1.3 * x[:, 0]), y)[0], 1.0, delta=0.05)


class EndToEndControlTests(unittest.TestCase):
    """The same machinery must find nothing in an efficient market and something when prices are blurred."""

    def _run(self, sd):
        m, _ = data.clean(synth.synth_raw(n_matches=20000, noise_sd=sd, seed=7), "ATP")
        f = pipeline.forecast_frame(data.orient(m), test_start="2010-01-01")
        return pipeline.test_sample(f, "2010-01-01")

    def test_no_edge_when_market_is_efficient(self):
        s = self._run(0.0)
        gap = market.paired_gap(market.log_loss(s["pB"], s["y"]), market.log_loss(s["ps_p1"], s["y"]))
        self.assertGreater(gap[1] - gap[2], 0)             # market clearly better than the simple model
        bet, profit, *_ = market.settle(s["pB"], s["max_o1"], s["max_o2"], s["y"], 0.05)
        self.assertLess(profit[bet].mean(), 0)

    def test_edge_found_when_prices_are_noisy(self):
        s = self._run(0.4)
        bet, profit, *_ = market.settle(s["pB"], s["max_o1"], s["max_o2"], s["y"], 0.05)
        n, roi, moe = market.mean_moe(profit[bet])
        self.assertGreater(roi - moe, 0)


class ReportTests(unittest.TestCase):
    def test_full_report_runs_on_synthetic_data(self):
        m, d = [], []
        for tour, seed in (("ATP", 11), ("WTA", 12)):
            mm, dd = data.clean(synth.synth_raw(n_matches=9000, seed=seed), tour)
            dd.update(file="x", tour=tour, year=0, kept=len(mm))
            m.append(mm)
            d.append(dd)
        text = report.build_report(pd.concat(m, ignore_index=True), d, "2010-01-01", with_controls=False)
        for head in ("1. DATA", "2. SANITY", "3. FORECAST", "4. CALIBRATION", "5. BETTING", "6. WHERE", "8. HEADLINE", "MULTIPLE TESTING"):
            self.assertIn(head, text)

    def test_z_crit(self):
        self.assertAlmostEqual(report.z_crit(0.05), 1.96, places=2)


if __name__ == "__main__":
    unittest.main()
