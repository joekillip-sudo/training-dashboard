"""Walk-forward backtest of the forecast model against Kalshi's own prices.

Time rules (the heart of "no future information"):
  * Two decision times per event day D (all in local STANDARD time, the settlement clock):
      "D-1 12:00": price = last hourly quote ending <= D-1 12:00 ; forecast = archived 'previous day 2'
      "D 06:00"  : price = last hourly quote ending <= D 06:00   ; forecast = archived 'previous day 1'
  * A forecast for hour h of day D at lead n comes from a model run started no later than (D - n days) at h, and is
    assumed usable only LATENCY hours later. We only use hours <= WINDOW_END, and test that this is <= decision time.
  * Bias/spread are fitted on days <= D-2 only (the climate report for D-1 isn't final at the decision time).
"""
import datetime as dt
import glob
import gzip
import math

import numpy as np
import pandas as pd

from .brackets import bounds, event_date, label_bounds
from .cities import CITIES
from .fees import fee_per_contract
from .model import MIN_TRAIN, bracket_prob, fit_bias_sigma

LEADS = {"D-1 12:00": dict(day_offset=-1, hour=12, fcst_lead=2),
         "D 06:00": dict(day_offset=0, hour=6, fcst_lead=1)}
LATENCY_H = 9          # model run start -> data usable (generous; ECMWF open data is the slow one)
WINDOW_START, WINDOW_END = 6, 21   # forecast daily max taken over these local-standard hours only
TRAIN_DAYS = 120
LAST_TRAIN_LAG = 2     # train on days <= D-2
CLIP = 0.005


# ---------------------------------------------------------------- time rules
def asof_epoch(day, lead, lst):
    """Epoch seconds of the decision time for event day `day`; local standard hour -> UTC via the fixed offset."""
    c = LEADS[lead]
    local = dt.datetime.combine(day + dt.timedelta(days=c["day_offset"]), dt.time(c["hour"]))
    return int(local.replace(tzinfo=dt.timezone.utc).timestamp()) - lst * 3600


def latest_info_epoch(day, fcst_lead, lst):
    """Worst-case moment the forecast used for `day` became available: run start for the LAST hour in the window,
    (day - fcst_lead days) at WINDOW_END local, plus processing latency."""
    start = dt.datetime.combine(day - dt.timedelta(days=fcst_lead), dt.time(WINDOW_END))
    return int(start.replace(tzinfo=dt.timezone.utc).timestamp()) - lst * 3600 + LATENCY_H * 3600


def train_cutoff(day):
    return day - dt.timedelta(days=LAST_TRAIN_LAG)


def quote_asof(q, ts):
    """q: DataFrame for one ticker with column 'ts'. Last quote ending at or before ts, else None. Never looks ahead."""
    q = q[q["ts"] <= ts]
    return None if q.empty else q.sort_values("ts").iloc[-1]


# ---------------------------------------------------------------- loading
def _read(pattern, **kw):
    fs = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f, **kw) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def load(d):
    mk = _read(f"{d}/*_markets.csv.gz")
    qt = _read(f"{d}/*_quotes.csv.gz")
    fc = _read(f"{d}/*_forecasts.csv.gz")
    ac = _read(f"{d}/*_actuals.csv")
    mk["date"] = mk["event"].map(event_date)
    fc["date"] = pd.to_datetime(fc["date"]).dt.date
    ac["date"] = pd.to_datetime(ac["date"]).dt.date
    return mk, qt, fc, ac


def prepare_events(mk):
    """One row per market with inclusive bounds. The label is the contract text, so it wins; the structured strikes
    are the cross-check (disagreements are returned for the report). Events must tile the line with exactly one winner."""
    rows, mism = [], 0
    for r in mk.itertuples():
        try:
            sb = bounds(r.strike_type, r.floor, r.cap)
        except (ValueError, TypeError):
            continue
        lb = label_bounds(r.subtitle)
        if lb is None:
            continue
        if lb != sb:
            mism += 1
        rows.append((r.ticker, r.event, r.city, r.date, lb[0], lb[1], r.result, r.volume, r.oi, r.open_time))
    df = pd.DataFrame(rows, columns=["ticker", "event", "city", "date", "lo", "hi", "result", "volume", "oi", "open_time"])
    ok = []
    for ev, g in df.groupby("event"):
        g = g.sort_values("lo")
        tiles = (g["lo"].iloc[0] == -math.inf and g["hi"].iloc[-1] == math.inf and
                 all(g["hi"].iloc[i] + 1 == g["lo"].iloc[i + 1] for i in range(len(g) - 1)))
        if tiles and (g["result"] == "yes").sum() == 1:
            ok.append(ev)
    return df[df["event"].isin(ok)].copy(), mism, df["event"].nunique(), len(ok)


# ---------------------------------------------------------------- forecast features
def day_table(fc, ac, city, lead, modelset):
    """date -> (x forecast daily max, y actual high). modelset 'gfs' or 'multi' (mean of all 4, all required)."""
    f = fc[(fc.city == city) & (fc.lead == LEADS[lead]["fcst_lead"])]
    if modelset == "gfs":
        x = f[f.model == "gfs_seamless"].set_index("date")["tmax"]
    else:
        p = f.pivot_table(index="date", columns="model", values="tmax")
        x = p.dropna().mean(axis=1) if p.shape[1] >= 4 else pd.Series(dtype=float)
    y = ac[ac.city == city].set_index("date")["high"]
    t = pd.DataFrame({"x": x}).join(y.rename("y"), how="left")
    return t


def predict_day(t, day):
    """(mu, sigma, n_train) for `day` using only rows with date <= day-2 inside the trailing window; None if unavailable."""
    if day not in t.index or pd.isna(t.loc[day, "x"]):
        return None
    cut = train_cutoff(day)
    tr = t[(t.index <= cut) & (t.index > cut - dt.timedelta(days=TRAIN_DAYS))].dropna()
    fit = fit_bias_sigma(tr["x"].to_numpy(), tr["y"].to_numpy())
    if fit is None:
        return None
    b, s = fit
    return t.loc[day, "x"] + b, s, len(tr)


# ---------------------------------------------------------------- the backtest
def build(mk, qt, fc, ac, modelsets=("gfs", "multi")):
    ev, mism, n_all, n_ok = prepare_events(mk)
    qgroups = {k: g for k, g in qt.groupby("ticker")}
    tables = {}
    out = []
    for (event, city), g in ev.groupby(["event", "city"]):
        day = g["date"].iloc[0]
        lst = CITIES[city]["lst"]
        winner = g.loc[g.result == "yes"].iloc[0]
        for lead in LEADS:
            ts = asof_epoch(day, lead, lst)
            quotes = {}
            for r in g.itertuples():
                if r.ticker in qgroups:
                    q = quote_asof(qgroups[r.ticker], ts)
                    if q is not None and q["ts"] > 0:
                        quotes[r.ticker] = q
            if len(quotes) < len(g):
                continue  # need a quote on every bracket to form the market's distribution
            for ms in modelsets:
                key = (city, lead, ms)
                if key not in tables:
                    tables[key] = day_table(fc, ac, city, lead, ms)
                pred = predict_day(tables[key], day)
                if pred is None:
                    continue
                mu, sigma, ntr = pred
                for r in g.itertuples():
                    q = quotes[r.ticker]
                    out.append(dict(event=event, city=city, date=day, lead=lead, modelset=ms, ticker=r.ticker,
                                    lo=r.lo, hi=r.hi, won=int(r.result == "yes"), mu=mu, sigma=sigma, n_train=ntr,
                                    p_model=bracket_prob(r.lo, r.hi, mu, sigma),
                                    bid=float(q["bid"]), ask=float(q["ask"]), last=q["last"], qvol=q["vol"], qoi=q["oi"],
                                    volume=r.volume, oi=r.oi, y=tables[key].loc[day, "y"]))
    df = pd.DataFrame(out)
    if not df.empty:
        df["mid"] = (df.bid + df.ask) / 2
        tot = df.groupby(["event", "lead", "modelset"])["mid"].transform("sum")
        df["sum_mid"] = tot
        df["p_market"] = df["mid"] / tot
    return df, dict(label_mismatch=mism, events_all=n_all, events_ok=n_ok)


# ---------------------------------------------------------------- scoring
def score(df, col):
    """Per-event log score of the winning bracket and Brier over all brackets for probability column `col`."""
    p = df[col].clip(CLIP, 1 - CLIP)
    d = df.assign(p=p)
    ls = d[d.won == 1].groupby(["event", "lead", "modelset"])["p"].apply(lambda s: float(np.log(s.iloc[0])))
    br = d.assign(se=(d.p - d.won) ** 2).groupby(["event", "lead", "modelset"])["se"].sum()
    return pd.DataFrame({"logscore": ls, "brier": br})


# ---------------------------------------------------------------- betting simulation
def simulate(df, edge=0.05, max_spread=0.10, min_oi=1.0, lots=None):
    """Flat 1-contract taker trades at the quoted ask (YES) or at 1-bid (NO). A trade needs: a two-sided book
    (bid>=0.01, ask<=0.99), spread <= max_spread, open interest > 0 at the decision time, and expected profit after the
    fee >= `edge`. Cost = price paid; pnl = payout - price - fee."""
    d = df[(df.bid >= 0.01) & (df.ask <= 0.99) & (df.ask - df.bid <= max_spread + 1e-9) & (df.qoi.fillna(0) >= min_oi)]
    rows = []
    for r in d.itertuples():
        for side in ("yes", "no"):
            price = r.ask if side == "yes" else 1 - r.bid
            p = r.p_model if side == "yes" else 1 - r.p_model
            fee = fee_per_contract(price, lots)
            if p - price - fee >= edge:
                win = r.won if side == "yes" else 1 - r.won
                rows.append(dict(event=r.event, city=r.city, date=r.date, lead=r.lead, modelset=r.modelset,
                                 ticker=r.ticker, side=side, price=price, p=p, fee=fee, win=win,
                                 pnl=win - price - fee, volume=r.volume))
    return pd.DataFrame(rows)
