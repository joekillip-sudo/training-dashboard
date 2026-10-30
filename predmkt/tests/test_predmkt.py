import datetime as dt
import math
import unittest

import numpy as np
import pandas as pd

from predmkt import backtest as bt
from predmkt.brackets import bounds, event_date, label_bounds
from predmkt.cities import CITIES
from predmkt.fees import fee_per_contract
from predmkt.model import bracket_prob, fit_bias_sigma, pit


def synth(n_days=400, sigma=3.0, seed=1, start=dt.date(2024, 1, 1)):
    rng = np.random.default_rng(seed)
    days = [start + dt.timedelta(days=i) for i in range(n_days)]
    truth = 60 + 20 * np.sin(np.arange(n_days) / 58) + rng.normal(0, 4, n_days)
    fc = truth + 1.5 + rng.normal(0, sigma, n_days)            # forecast = truth + bias 1.5 + noise
    y = np.round(truth)
    return days, fc, y


def table(days, fc, y):
    return pd.DataFrame({"x": fc, "y": y}, index=days)


class BracketTests(unittest.TestCase):
    def test_event_date(self):
        self.assertEqual(event_date("KXHIGHNY-26OCT29"), dt.date(2026, 10, 29))
        self.assertIsNone(event_date("KXHIGHNY"))

    def test_bounds_and_labels_agree(self):
        self.assertEqual(bounds("greater", 74, None), (75.0, math.inf))
        self.assertEqual(label_bounds("75° or above"), (75.0, math.inf))
        self.assertEqual(label_bounds("63° to 64°"), bounds("between", 63, 64))
        self.assertEqual(label_bounds("58° or below")[1], 58.0)

    def test_probabilities_tile_to_one(self):
        edges = [(-math.inf, 59), (60, 61), (62, 63), (64, 65), (66, math.inf)]
        s = sum(bracket_prob(lo, hi, 63.2, 2.5) for lo, hi in edges)
        self.assertAlmostEqual(s, 1.0, places=9)


class FeeTests(unittest.TestCase):
    def test_fee(self):
        self.assertAlmostEqual(fee_per_contract(0.5), 0.0175)
        self.assertAlmostEqual(fee_per_contract(0.5, lots=100), 0.0175)
        self.assertAlmostEqual(fee_per_contract(0.5, lots=1), 0.02)   # rounded up to the cent
        self.assertAlmostEqual(fee_per_contract(0.01), 0.07 * 0.01 * 0.99)


class TimingTests(unittest.TestCase):
    def test_forecast_is_available_before_decision_time(self):
        d = dt.date(2025, 7, 10)
        for city, c in CITIES.items():
            for lead, cfg in bt.LEADS.items():
                self.assertLessEqual(bt.latest_info_epoch(d, cfg["fcst_lead"], c["lst"]),
                                     bt.asof_epoch(d, lead, c["lst"]), (city, lead))

    def test_asof_epoch_known_value(self):
        # New York (UTC-5 standard): 2026-10-28 12:00 local standard = 17:00Z
        e = bt.asof_epoch(dt.date(2026, 10, 29), "D-1 12:00", -5)
        self.assertEqual(dt.datetime.fromtimestamp(e, dt.timezone.utc), dt.datetime(2026, 10, 28, 17, tzinfo=dt.timezone.utc))

    def test_quote_asof_ignores_the_future(self):
        q = pd.DataFrame({"ts": [100, 200, 300], "bid": [0.1, 0.2, 0.99], "ask": [0.2, 0.3, 1.0]})
        self.assertEqual(bt.quote_asof(q, 250)["bid"], 0.2)
        self.assertIsNone(bt.quote_asof(q, 50))

    def test_training_never_sees_recent_or_future_days(self):
        days, fc, y = synth()
        t = table(days, fc, y)
        D = days[300]
        base = bt.predict_day(t, D)
        t2 = t.copy()
        for d in days:
            if d >= D - dt.timedelta(days=1):                       # corrupt D-1, D and everything after
                t2.loc[d, "y"] = 999.0
                if d != D:
                    t2.loc[d, "x"] = -999.0
        self.assertEqual(base, bt.predict_day(t2, D))
        # but it does use D-2
        t3 = t.copy()
        t3.loc[D - dt.timedelta(days=2), "y"] += 50
        self.assertNotEqual(base, bt.predict_day(t3, D))


class ModelSanityTests(unittest.TestCase):
    def test_fit_recovers_bias_and_scale_once(self):
        days, fc, y = synth(n_days=2000, sigma=3.0)
        b, s = fit_bias_sigma(fc * 0 + (y - 0), y)  # zero residual -> floor applies once
        self.assertEqual(s, 1.0)
        # forecast has +1.5 error vs truth => actual - forecast has mean ~ -1.5 and sd ~ sqrt(3^2 + rounding)
        b, s = fit_bias_sigma(fc, y)
        self.assertAlmostEqual(b, -1.5, delta=0.25)
        self.assertAlmostEqual(s, 3.0, delta=0.3)   # a scale applied twice would give ~9 or ~1

    def test_pit_is_uniform_and_z_has_unit_sd_out_of_sample(self):
        days, fc, y = synth(n_days=900, sigma=3.0)
        t = table(days, fc, y)
        us, zs = [], []
        rng = np.random.default_rng(0)
        for d in days[150:]:
            mu, s, _ = bt.predict_day(t, d)
            us.append(pit(t.loc[d, "y"], mu, s, rng.random()))
            zs.append((t.loc[d, "y"] - mu) / s)
        self.assertAlmostEqual(float(np.std(zs)), 1.0, delta=0.12)
        self.assertAlmostEqual(float(np.mean(us)), 0.5, delta=0.05)
        h = np.histogram(us, bins=5, range=(0, 1))[0] / len(us)
        self.assertTrue(all(0.14 < x < 0.26 for x in h), h)


def toy_world(n_days=260):
    """A tiny market world: Kalshi-style brackets on an unbiased forecast, prices = the true probabilities."""
    days, fc, y = synth(n_days=n_days, sigma=3.0, start=dt.date(2024, 1, 1))
    rows_m, rows_q, rows_f, rows_a = [], [], [], []
    edges = [(-math.inf, 55), (56, 57), (58, 59), (60, 61), (62, 63), (64, math.inf)]
    for i, d in enumerate(days):
        ev = "KXHIGHNY-" + d.strftime("%y%b%d").upper()
        for j, (lo, hi) in enumerate(edges):
            sub = ("%d° or below" % hi if lo == -math.inf else "%d° or above" % lo if hi == math.inf
                   else "%d° to %d°" % (lo, hi))
            st = "less" if lo == -math.inf else "greater" if hi == math.inf else "between"
            fl = lo - 1 if st == "greater" else lo
            cp = hi + 1 if st == "less" else hi
            win = lo <= y[i] <= hi
            tk = f"{ev}-{j}"
            rows_m.append(dict(ticker=tk, event=ev, city="NYC", strike_type=st, floor=None if st == "less" else fl,
                               cap=None if st == "greater" else cp, subtitle=sub, result="yes" if win else "no",
                               exp_value="", open_time="", close_time="", volume=1000, oi=500))
            p = min(max(bracket_prob(lo, hi, y[i] * 0 + 60, 5.0), 0.02), 0.95)
            for lead in bt.LEADS:
                ts = bt.asof_epoch(d, lead, -5)
                rows_q.append(dict(ticker=tk, ts=ts - 3600, bid=round(p - 0.02, 2), ask=round(p + 0.02, 2),
                                   last=p, vol=5, oi=100))
        for lead_n in (1, 2):
            rows_f.append(dict(city="NYC", date=d, model="gfs_seamless", lead=lead_n, tmax=fc[i], n_hours=16))
        rows_a.append(dict(city="NYC", date=d, high=y[i]))
    mk = pd.DataFrame(rows_m)
    mk["date"] = mk["event"].map(event_date)
    return mk, pd.DataFrame(rows_q), pd.DataFrame(rows_f), pd.DataFrame(rows_a), days


class PipelineTests(unittest.TestCase):
    def test_no_future_information_in_the_built_dataset(self):
        mk, qt, fc, ac, days = toy_world()
        df, meta = bt.build(mk, qt, fc, ac, modelsets=("gfs",))
        self.assertFalse(df.empty)
        self.assertEqual(meta["events_all"], meta["events_ok"])
        D = days[200]
        base = df[(df.date == D) & (df.lead == "D 06:00")].sort_values("ticker")
        # add quotes AFTER the decision time and rewrite actuals/forecasts for D-1, D and later
        ts = bt.asof_epoch(D, "D 06:00", -5)
        qt2 = pd.concat([qt, pd.DataFrame([dict(ticker=t, ts=ts + 3600, bid=0.01, ask=0.02, last=0.5, vol=1, oi=1)
                                           for t in mk[mk.date == D].ticker])])
        ac2 = ac.copy()
        ac2.loc[ac2.date >= D - dt.timedelta(days=1), "high"] = 120
        fc2 = fc.copy()
        fc2.loc[(fc2.date > D), "tmax"] = -50
        df2, _ = bt.build(mk, qt2, fc2, ac2, modelsets=("gfs",))
        new = df2[(df2.date == D) & (df2.lead == "D 06:00")].sort_values("ticker")
        for col in ("p_model", "bid", "ask", "mu", "sigma"):
            np.testing.assert_allclose(base[col].to_numpy(), new[col].to_numpy(), err_msg=col)

    def test_model_probabilities_sum_to_one_per_event(self):
        mk, qt, fc, ac, _ = toy_world()
        df, _ = bt.build(mk, qt, fc, ac, modelsets=("gfs",))
        s = df.groupby(["event", "lead"])["p_model"].sum()
        np.testing.assert_allclose(s.to_numpy(), 1.0, atol=1e-9)

    def test_no_edge_world_has_no_profitable_bets_after_fees(self):
        # prices are the (wrong-by-design) climatology; the model has real skill here, so we only check mechanics:
        mk, qt, fc, ac, _ = toy_world()
        df, _ = bt.build(mk, qt, fc, ac, modelsets=("gfs",))
        bets = bt.simulate(df)
        self.assertTrue((bets.pnl == bets.win - bets.price - bets.fee).all())
        self.assertTrue((bets.price > 0).all() and (bets.price < 1).all())

    def test_score_prefers_the_sharper_correct_forecast(self):
        d = pd.DataFrame({"event": ["e"] * 3, "lead": ["l"] * 3, "modelset": ["m"] * 3, "won": [0, 1, 0],
                          "a": [0.1, 0.8, 0.1], "b": [0.4, 0.2, 0.4]})
        self.assertGreater(bt.score(d, "a")["logscore"].iloc[0], bt.score(d, "b")["logscore"].iloc[0])
        self.assertLess(bt.score(d, "a")["brier"].iloc[0], bt.score(d, "b")["brier"].iloc[0])


if __name__ == "__main__":
    unittest.main()


class QuoteTests(unittest.TestCase):
    def test_real_quote_rejects_empty_books(self):
        self.assertFalse(bt.real_quote(pd.Series({"bid": 0.0, "ask": 0.99})))
        self.assertFalse(bt.real_quote(pd.Series({"bid": 0.0, "ask": 1.0})))
        self.assertTrue(bt.real_quote(pd.Series({"bid": 0.0, "ask": 0.07})))
        self.assertTrue(bt.real_quote(pd.Series({"bid": 0.42, "ask": 0.46})))
