"""Download one city's data (run on a GitHub Actions runner): markets, hourly quotes, archived forecasts, truth.

    python -m predmkt.fetch NYC --out predmkt/data_raw
Writes <out>/<CITY>_{markets,quotes,forecasts,actuals}.csv.gz  (small, compressed)
"""
import csv
import datetime as dt
import gzip
import os
import sys

from .brackets import event_date
from .cities import CITIES, MODELS
from .http import get

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
STEP_DAYS = 2  # keep every 2nd event day (by ordinal): halves calls and repo size, unbiased w.r.t. weather


def _num(d, *keys):
    """First present key among alternatives (live tier says close_dollars, historical says close)."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return ""


def list_markets(series):
    out = {}
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cur, pages = None, 0
        while pages < 80:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            s, j, _ = get(KALSHI + path, p, pause=0.15)
            if s != 200 or not isinstance(j, dict):
                print(f"  list {path} status {s}", flush=True)
                break
            for m in j.get("markets", []):
                out[m["ticker"]] = m
            cur, pages = j.get("cursor"), pages + 1
            if not cur:
                break
    return out


def epoch(s):
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def candles(m, series, cutoff):
    p = {"start_ts": epoch(m["open_time"]), "end_ts": epoch(m["close_time"]), "period_interval": 60}
    hist_first = m["close_time"][:10] < cutoff
    paths = [f"/historical/markets/{m['ticker']}/candlesticks", f"/series/{series}/markets/{m['ticker']}/candlesticks"]
    if not hist_first:
        paths.reverse()
    for path in paths:
        s, j, _ = get(KALSHI + path, p, pause=0.15)
        if s == 200 and isinstance(j, dict) and j.get("candlesticks") is not None:
            return j["candlesticks"]
    return None


def run_kalshi(city, out):
    c = CITIES[city]
    s, j, _ = get(KALSHI + "/historical/cutoff")
    cutoff = (j.get("market_settled_ts") or "2026-08-31")[:10] if isinstance(j, dict) else "2026-08-31"
    mk = list_markets(c["series"])
    print(f"{city}: {len(mk)} markets listed", flush=True)
    keep = [m for m in mk.values() if event_date(m["event_ticker"]) and event_date(m["event_ticker"]).toordinal() % STEP_DAYS == 0
            and m.get("open_time") and m.get("close_time")]
    keep.sort(key=lambda m: m["close_time"])
    print(f"{city}: {len(keep)} markets in the every-{STEP_DAYS}nd-day sample", flush=True)
    mf = gzip.open(f"{out}/{city}_markets.csv.gz", "wt", newline="")
    qf = gzip.open(f"{out}/{city}_quotes.csv.gz", "wt", newline="")
    mw, qw = csv.writer(mf), csv.writer(qf)
    mw.writerow(["ticker", "event", "city", "strike_type", "floor", "cap", "subtitle", "result", "exp_value",
                 "open_time", "close_time", "volume", "oi"])
    qw.writerow(["ticker", "ts", "bid", "ask", "last", "vol", "oi"])
    bad = 0
    for i, m in enumerate(keep):
        cs = candles(m, c["series"], cutoff)
        if cs is None:
            bad += 1
            continue
        mw.writerow([m["ticker"], m["event_ticker"], city, m.get("strike_type"), m.get("floor_strike"),
                     m.get("cap_strike"), m.get("yes_sub_title"), m.get("result"), m.get("expiration_value"),
                     m["open_time"], m["close_time"], _num(m, "volume_fp", "volume"),
                     _num(m, "open_interest_fp", "open_interest")])
        for k in cs:
            qw.writerow([m["ticker"], k.get("end_period_ts"),
                         _num(k.get("yes_bid"), "close_dollars", "close"), _num(k.get("yes_ask"), "close_dollars", "close"),
                         _num(k.get("price"), "close_dollars", "close"), _num(k, "volume_fp", "volume"),
                         _num(k, "open_interest_fp", "open_interest")])
        if i % 500 == 0:
            print(f"  {city} {i}/{len(keep)} (no-candle markets so far {bad})", flush=True)
    mf.close()
    qf.close()
    print(f"{city}: done, {bad} markets without candles", flush=True)
    return keep


def run_forecasts(city, out, first_day):
    """Hourly previous-day-1 / previous-day-2 forecasts -> daily max over 06-21 local STANDARD time."""
    c = CITIES[city]
    start = max(dt.date(2021, 6, 1), first_day - dt.timedelta(days=150))
    end = dt.date.today() - dt.timedelta(days=1)
    rows = {}
    chunk = start
    while chunk <= end:
        stop = min(chunk + dt.timedelta(days=180), end)
        hv = ",".join(f"temperature_2m_previous_day{n}" for n in (1, 2))
        s, j, _ = get("https://previous-runs-api.open-meteo.com/v1/forecast",
                      {"latitude": c["lat"], "longitude": c["lon"], "start_date": chunk.isoformat(),
                       "end_date": (stop + dt.timedelta(days=1)).isoformat(), "hourly": hv,
                       "models": ",".join(MODELS), "temperature_unit": "fahrenheit", "timezone": "UTC"}, pause=1.0)
        if s != 200:
            print(f"  forecasts {chunk} status {s}: {str(j)[:200]}", flush=True)
            chunk = stop + dt.timedelta(days=1)
            continue
        hr = j["hourly"]
        times = [dt.datetime.fromisoformat(t) for t in hr["time"]]
        for model in MODELS:
            for lead in (1, 2):
                vals = hr.get(f"temperature_2m_previous_day{lead}_{model}") or hr.get(f"temperature_2m_previous_day{lead}")
                if vals is None:
                    continue
                for t, v in zip(times, vals):
                    if v is None:
                        continue
                    lst = t + dt.timedelta(hours=c["lst"])
                    if 6 <= lst.hour <= 21:
                        rows.setdefault((lst.date(), model, lead), []).append(v)
        chunk = stop + dt.timedelta(days=1)
    with gzip.open(f"{out}/{city}_forecasts.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "date", "model", "lead", "tmax", "n_hours"])
        for (d, model, lead), v in sorted(rows.items()):
            if len(v) >= 14:  # need nearly the full 16-hour window
                w.writerow([city, d.isoformat(), model, lead, round(max(v), 2), len(v)])
    print(f"{city}: forecast rows for {len(rows)} (day,model,lead)", flush=True)


def run_actuals(city, out, first_year):
    """Official daily high from the NWS climate report (CLI) via Iowa Environmental Mesonet."""
    c = CITIES[city]
    n = 0
    with open(f"{out}/{city}_actuals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "date", "high"])
        for y in range(first_year, dt.date.today().year + 1):
            s, j, _ = get("https://mesonet.agron.iastate.edu/json/cli.py", {"station": c["iem"], "year": y}, pause=1.0)
            if s != 200 or not isinstance(j, dict):
                print(f"  actuals {y} status {s}", flush=True)
                continue
            for r in j.get("results", []):
                if r.get("high") is not None:
                    w.writerow([city, r["valid"], r["high"]])
                    n += 1
    print(f"{city}: {n} actual daily highs", flush=True)


if __name__ == "__main__":
    city = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "predmkt/data_raw"
    os.makedirs(out, exist_ok=True)
    keep = run_kalshi(city, out)
    days = sorted(event_date(m["event_ticker"]) for m in keep)
    run_forecasts(city, out, days[0])
    run_actuals(city, out, days[0].year - 1)
