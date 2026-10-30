"""Build predmkt_report.txt from the fetched data.   python -m predmkt.report --data predmkt/data --out predmkt_report.txt"""
import argparse
import math
import sys

import numpy as np
import pandas as pd

from . import backtest as bt
from .fetch import STEP_DAYS
from .model import cdf_le, fit_bias_sigma, pit

PRIMARY = dict(edge=0.05, max_spread=0.10)
CAPTURE = 0.05            # share of a bracket's lifetime volume we assume we could take without moving price
CAPACITY_BAR = 10_000     # $ per year of profit, below which we call the opportunity trivial
rng = np.random.default_rng(12345)


def season(m):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA"}.get(m, "SON")


def mean_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    return float(x.mean()), 1.96 * float(x.std(ddof=1) / math.sqrt(len(x)))


def roi_ci(b, n=2000):
    """ROI = total pnl / total price paid, with an event-level bootstrap 95% interval."""
    if b.empty:
        return float("nan"), float("nan"), float("nan")
    g = b.groupby("event").agg(pnl=("pnl", "sum"), cost=("price", "sum"))
    pnl, cost = g.pnl.to_numpy(), g.cost.to_numpy()
    idx = rng.integers(0, len(g), (n, len(g)))
    r = pnl[idx].sum(1) / cost[idx].sum(1)
    return pnl.sum() / cost.sum(), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def fmt_roi(b):
    if len(b) == 0:
        return "no trades"
    r, lo, hi = roi_ci(b)
    return f"n={len(b):5d} ROI {r*100:+6.1f}%  95% CI [{lo*100:+6.1f}%, {hi*100:+6.1f}%]  hit {b.win.mean()*100:4.1f}%"


def halves(b):
    if b.empty:
        return None, None
    cut = b["date"].sort_values().iloc[len(b) // 2]
    return b[b.date < cut], b[b.date >= cut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="predmkt/data")
    ap.add_argument("--out", default="predmkt_report.txt")
    a = ap.parse_args()
    mk, qt, fc, ac = bt.load(a.data)
    L = []
    P = L.append
    P("PREDICTION-MARKET WEATHER EDGE TEST  (paper analysis only; Kalshi daily-high temperature markets)")
    P("=" * 100)
    ev, mism, n_all, n_ok = bt.prepare_events(mk)
    P(f"\n1. DATA\nmarkets loaded {len(mk)}  events {n_all}  events that tile cleanly with exactly one winner {n_ok}")
    P(f"bracket label vs structured-strike disagreements (label used): {mism}")
    P(f"sampling: every {STEP_DAYS}nd calendar day (by ordinal) - unbiased with respect to weather, halves the data")
    P(ev.groupby("city").agg(events=("event", "nunique"), first=("date", "min"), last=("date", "max")).to_string())
    df, meta = bt.build(mk, qt, fc, ac)
    if df.empty:
        P("\nNo scoreable events. Stopping."); open(a.out, "w").write("\n".join(L) + "\n"); return
    df["season"] = pd.to_datetime(df["date"]).dt.month.map(season)
    # truth consistency: official CLI high must fall inside the bracket Kalshi paid
    w = df[df.won == 1].drop_duplicates(["event"])
    ok = ((w.y >= w.lo) & (w.y <= w.hi))
    P(f"\ntruth cross-check: NWS climate-report high lies inside the bracket Kalshi actually paid in {ok.mean()*100:.1f}% of {len(w)} events")
    pe = df.drop_duplicates(["event", "lead", "modelset"])
    P("\nscoreable events (a quote on every bracket at the decision time AND enough training days):")
    P(pe.groupby(["lead", "modelset"]).agg(events=("event", "nunique"), first=("date", "min"), last=("date", "max")).to_string())
    P("\nquote sanity: median sum of bracket mids (should be ~1; the excess is the market's overround)")
    P(pe.groupby("lead")["sum_mid"].describe()[["count", "25%", "50%", "75%"]].round(3).to_string())

    P("\n\n2. DOES THE MODEL BEAT THE MARKET'S OWN FORECAST?  (higher log score / lower Brier = better; paired by event)")
    P("   market probability = mid-quote at the decision time, normalised to sum to 1 across brackets.")
    sm, sk = score(df, "p_model"), score(df, "p_market")
    d = sm.join(sk, lsuffix="_model", rsuffix="_market")
    d["dlog"] = d.logscore_model - d.logscore_market
    d["dbrier"] = d.brier_market - d.brier_model   # positive = model better
    for (lead, ms), g in d.groupby(level=["lead", "modelset"]):
        m, h = mean_ci(g.dlog)
        mb, hb = mean_ci(g.dbrier)
        P(f"  {lead:10s} {ms:6s} events {len(g):5d}  logscore model {g.logscore_model.mean():+.3f} market {g.logscore_market.mean():+.3f}  "
          f"diff {m:+.3f} +- {h:.3f}   Brier model {g.brier_model.mean():.3f} market {g.brier_market.mean():.3f}  advantage {mb:+.3f} +- {hb:.3f}")
    P("\n  by city (primary lead 'D 06:00', model set 'gfs'; diff = model - market log score):")
    dd = d.reset_index().merge(pe[["event", "city", "season"]].drop_duplicates("event"), on="event")
    for key in ("city", "season"):
        for k, g in dd[(dd.lead == "D 06:00") & (dd.modelset == "gfs")].groupby(key):
            m, h = mean_ci(g.dlog)
            P(f"    {k:5s} events {len(g):4d}  diff {m:+.3f} +- {h:.3f}")

    P("\n\n3. CALIBRATION BY PROBABILITY BAND (pooled over brackets; predicted vs how often the bracket actually won)")
    bins = [0, .02, .05, .1, .2, .35, .5, 1.0]
    for ms in ("gfs",):
        for lead in bt.LEADS:
            g = df[(df.modelset == ms) & (df.lead == lead)]
            P(f"  {lead} / {ms}: band | n | model p -> observed | market p -> observed")
            for lo, hi in zip(bins[:-1], bins[1:]):
                a1 = g[(g.p_model > lo) & (g.p_model <= hi)]
                a2 = g[(g.p_market > lo) & (g.p_market <= hi)]
                P(f"    ({lo:.2f},{hi:.2f}] | model n={len(a1):6d} {a1.p_model.mean():.3f} -> {a1.won.mean():.3f} | market n={len(a2):6d} {a2.p_market.mean():.3f} -> {a2.won.mean():.3f}"
                  if len(a1) and len(a2) else f"    ({lo:.2f},{hi:.2f}] | n/a")

    P("\n\n4. SELF-CHECK OF THE MODEL AGAINST ITS OWN TRAINING DATA (PIT / z-scores; the 'scale applied twice' test)")
    P("   z = (actual - predicted mean)/sigma should have mean 0, sd 1; PIT deciles should each hold ~10%.")
    for ms in ("gfs", "multi"):
        for lead in bt.LEADS:
            ins_z, ins_u, oos_z, oos_u = [], [], [], []
            for city in sorted(pe.city.unique()):
                t = bt.day_table(fc, ac, city, lead, ms).dropna()
                if len(t) < 60:
                    continue
                fit = fit_bias_sigma(t.x, t.y)
                b, s = fit
                for x, y in zip(t.x, t.y):          # in-sample: full-history fit applied to its own days
                    mu = x + b
                    ins_z.append((y - mu) / s)
                    ins_u.append(pit(y, mu, s, rng.random()))
            g = pe[(pe.modelset == ms) & (pe.lead == lead)]
            for r in g.itertuples():                 # out-of-sample: the walk-forward predictions actually used
                oos_z.append((r.y - r.mu) / r.sigma)
                oos_u.append(pit(r.y, r.mu, r.sigma, rng.random()))
            if not ins_z or not oos_z:
                continue
            hi_ = np.histogram(ins_u, bins=10, range=(0, 1))[0] / len(ins_u)
            ho_ = np.histogram(oos_u, bins=10, range=(0, 1))[0] / len(oos_u)
            P(f"  {lead:10s} {ms:5s} in-sample: n={len(ins_z)} z mean {np.mean(ins_z):+.2f} sd {np.std(ins_z):.2f} | out-of-sample: n={len(oos_z)} z mean {np.mean(oos_z):+.2f} sd {np.std(oos_z):.2f}")
            P(f"      PIT deciles in-sample    {' '.join(f'{x*100:4.1f}' for x in hi_)}")
            P(f"      PIT deciles out-of-sample {' '.join(f'{x*100:4.1f}' for x in ho_)}")

    P("\n\n5. BETTING SIMULATION  (1 contract per trade, taker at the quoted ask / 1-bid, Kalshi fee 0.07*P*(1-P) per contract)")
    P(f"   primary spec fixed in advance: expected profit after fee >= {PRIMARY['edge']:.2f}, spread <= {PRIMARY['max_spread']:.2f}, open interest > 0.")
    P("   fill assumption: you can buy 1 contract at the last hourly quote before the decision time. Order-book depth is NOT")
    P("   available historically, so this is optimistic for anything larger than a few contracts.")
    prim = bt.simulate(df, **PRIMARY)
    nslices = 0
    for (lead, ms), g in prim.groupby(["lead", "modelset"]):
        h1, h2 = halves(g)
        P(f"\n  {lead} / {ms}: ALL    {fmt_roi(g)}")
        P(f"      first half  {fmt_roi(h1)}   [{h1.date.min()} .. {h1.date.max()}]")
        P(f"      second half {fmt_roi(h2)}   [{h2.date.min()} .. {h2.date.max()}]")
        P(f"      avg fee {g.fee.mean()*100:.2f}c/contract, avg price {g.price.mean():.3f}, share of trades buying NO {np.mean(g.side=='no')*100:.0f}%")
    P("\n  sensitivity grid (every cell is another look at the same data - multiple testing; do not pick the best one):")
    for edge in (0.02, 0.05, 0.10):
        for spread in (0.05, 0.10, 0.20):
            for lots in (None, 1):
                b = bt.simulate(df, edge=edge, max_spread=spread, lots=lots)
                for (lead, ms), g in b.groupby(["lead", "modelset"]):
                    nslices += 1
                    r, lo, hi = roi_ci(g)
                    P(f"    edge>={edge:.2f} spread<={spread:.2f} fees={'1-lot' if lots else 'large'} {lead:10s} {ms:5s} n={len(g):5d} ROI {r*100:+6.1f}% [{lo*100:+6.1f},{hi*100:+6.1f}]")
    P(f"\n  breakdown of the primary spec by city / season (D 06:00, gfs, then D-1 12:00, gfs):")
    for lead in bt.LEADS:
        g0 = prim[(prim.lead == lead) & (prim.modelset == "gfs")].copy()
        if g0.empty:
            continue
        g0["season"] = pd.to_datetime(g0["date"]).dt.month.map(season)
        for key in ("city", "season"):
            for k, g in g0.groupby(key):
                P(f"    {lead:10s} {key:6s}={k:4s} {fmt_roi(g)}")

    P("\n\n6. CAPACITY")
    P(f"   assume we could take at most {CAPTURE*100:.0f}% of a bracket's total traded volume without moving its price")
    P("   (book depth isn't archived; this is a rough ceiling, not a measurement).")
    years = max((pd.to_datetime(df.date).max() - pd.to_datetime(df.date).min()).days / 365.25, 0.1)
    for (lead, ms), g in prim.groupby(["lead", "modelset"]):
        vol = pd.to_numeric(g.volume, errors="coerce").fillna(0)
        size = CAPTURE * vol
        profit = (size * g.pnl).sum() * STEP_DAYS / years
        stake = (size * g.price).sum() * STEP_DAYS / years
        P(f"  {lead:10s} {ms:5s} trades/yr (all days) ~{len(g)*STEP_DAYS/years:7.0f}  median bracket volume {vol.median():,.0f} contracts  "
          f"capital turned/yr ~${stake:,.0f}  implied profit/yr ~${profit:,.0f}")
    allp = df.drop_duplicates("ticker")
    P(f"  context: median bracket lifetime volume {pd.to_numeric(allp.volume, errors='coerce').median():,.0f} contracts "
      f"(~${pd.to_numeric(allp.volume, errors='coerce').median()*0.5:,.0f} at ~50c); median open interest {pd.to_numeric(allp.oi, errors='coerce').median():,.0f}")

    P("\n\n7. VERDICT AGAINST THE PRE-STATED CRITERIA")
    P(f"   all must hold: (a) model log score beats market, CI excludes 0; (b) primary-spec ROI > 0 with 95% CI above 0;")
    P(f"   (c) both halves ROI > 0; (d) implied capacity profit >= ${CAPACITY_BAR:,}/yr.  ({nslices} sensitivity cells + many slices were viewed.)")
    any_pass = False
    for (lead, ms), g in prim.groupby(["lead", "modelset"]):
        dl = d.xs((lead, ms), level=("lead", "modelset"))
        m, h = mean_ci(dl.dlog)
        r, lo, hi = roi_ci(g)
        h1, h2 = halves(g)
        r1 = h1.pnl.sum() / h1.price.sum() if len(h1) else float("nan")
        r2 = h2.pnl.sum() / h2.price.sum() if len(h2) else float("nan")
        vol = pd.to_numeric(g.volume, errors="coerce").fillna(0)
        cap = (CAPTURE * vol * g.pnl).sum() * STEP_DAYS / years
        c = [m - h > 0, lo > 0, r1 > 0 and r2 > 0, cap >= CAPACITY_BAR]
        any_pass |= all(c)
        P(f"  {lead:10s} {ms:5s} (a){'PASS' if c[0] else 'fail'} (b){'PASS' if c[1] else 'fail'} (c){'PASS' if c[2] else 'fail'} (d){'PASS' if c[3] else 'fail'}")
    P("\n  OVERALL: " + ("at least one configuration meets every criterion - treat with suspicion and test forward in time."
                         if any_pass else "NO USABLE EDGE: no configuration meets all four criteria."))
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def score(df, col):
    return bt.score(df, col)


if __name__ == "__main__":
    main()
