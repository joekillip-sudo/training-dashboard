"""STEP 0, second pass: how deep is the history, and is the pricing/forecast data point-in-time?"""
import collections
import datetime as dt
import json
import statistics as st_

from .http import get
from .probe import KALSHI, POLY_CLOB, POLY_GAMMA, section, short


def ts(s):
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def kalshi_hist():
    section("KALSHI historical tier: depth by series")
    keep = {}
    for sr in ("KXHIGHNY", "HIGHNY", "KXHIGHCHI", "HIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHLAX", "KXHIGHDEN",
               "KXHIGHPHIL", "KXHIGHTSEA", "KXHIGHTATL", "KXHIGHTDAL", "KXHIGHTBOS", "KXHIGHTHOU", "KXHIGHTSFO"):
        mk, cur, pages = [], None, 0
        while pages < 60:
            p = {"series_ticker": sr, "limit": 1000}
            if cur:
                p["cursor"] = cur
            s, j, h = get(f"{KALSHI}/historical/markets", p)
            if s != 200 or not isinstance(j, dict):
                print(sr, "status", s, short(j, 200))
                break
            mk += j.get("markets", [])
            cur, pages = j.get("cursor"), pages + 1
            if not cur:
                break
        if not mk:
            print(sr, "-> 0 historical markets")
            continue
        ct = sorted(m.get("close_time", "") for m in mk)
        ev = {m.get("event_ticker") for m in mk}
        vols = sorted(f(m.get("volume_fp")) for m in mk)
        withval = sum(1 for m in mk if m.get("expiration_value") not in (None, ""))
        print(f"{sr}: {len(mk)} markets, {len(ev)} events, {ct[0][:10]} .. {ct[-1][:10]}, pages {pages}; "
              f"volume_fp median {vols[len(vols)//2]:.0f} p90 {vols[int(len(vols)*.9)]:.0f} max {vols[-1]:.0f}; "
              f"with expiration_value {withval}")
        bm = collections.Counter(c[:7] for c in ct)
        print("   by month:", dict(sorted(bm.items())))
        keep[sr] = mk
    mk = keep.get("KXHIGHNY") or next(iter(keep.values()), [])
    if mk:
        print("\nstrike types in KXHIGHNY:", collections.Counter(m.get("strike_type") for m in mk))
        old = sorted(mk, key=lambda m: m["close_time"])[len(mk) // 3]
        print("example old market:", short({k: old.get(k) for k in ("ticker", "title", "yes_sub_title", "open_time",
              "close_time", "expiration_value", "result", "floor_strike", "cap_strike", "strike_type", "volume_fp",
              "open_interest_fp")}, 900))
        t, sr = old["ticker"], old["ticker"].split("-")[0]
        section(f"KALSHI historical price/trade endpoints for {t}")
        for path, p in ((f"/historical/markets/{t}/candlesticks",
                         {"start_ts": ts(old["open_time"]), "end_ts": ts(old["close_time"]), "period_interval": 60}),
                        (f"/series/{sr}/markets/{t}/candlesticks",
                         {"start_ts": ts(old["open_time"]), "end_ts": ts(old["close_time"]), "period_interval": 60}),
                        ("/historical/trades", {"ticker": t, "limit": 1000}),
                        ("/markets/trades", {"ticker": t, "limit": 1000})):
            s, j, h = get(KALSHI + path, p)
            n = None
            if isinstance(j, dict):
                n = {k: len(v) for k, v in j.items() if isinstance(v, list)}
            print(path, "status", s, n, short(j, 350) if s != 200 or not n else "")
            if s == 200 and isinstance(j, dict):
                for k, v in j.items():
                    if isinstance(v, list) and v:
                        print("   first:", short(v[0], 500), "\n   last:", short(v[-1], 500))
    section("KALSHI: fee schedule")
    for path in ("/series/fee_changes", "/exchange/announcements"):
        s, j, h = get(KALSHI + path)
        print(path, s, short(j, 700))
    return keep


def pm_depth():
    section("POLYMARKET: how far back do daily temperature events go? (NYC, Chicago, Miami, London)")
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"]
    tok_example = None
    for city in ("nyc", "chicago", "miami", "london", "dallas", "seattle", "atlanta"):
        hits = []
        for (y, m, d) in ((2024, 6, 15), (2025, 1, 15), (2025, 4, 15), (2025, 7, 15), (2025, 10, 15),
                          (2026, 1, 15), (2026, 4, 15), (2026, 7, 15), (2026, 10, 15)):
            slug = f"highest-temperature-in-{city}-on-{months[m-1]}-{d}-{y}"
            s, j, h = get(f"{POLY_GAMMA}/events", {"slug": slug})
            if isinstance(j, list) and j:
                e = j[0]
                hits.append((f"{y}-{m:02d}-{d}", len(e.get("markets", [])), e.get("volume")))
                if tok_example is None and e.get("markets"):
                    tok_example = e
        print(city, hits)
    e = tok_example
    if e:
        print("\nexample event:", e.get("slug"), "vol", e.get("volume"), "liq", e.get("liquidity"))
        for m in e["markets"][:3]:
            print("  ", short({k: m.get(k) for k in ("question", "volume", "liquidity", "outcomePrices", "bestBid",
                  "bestAsk", "spread", "resolutionSource", "feesEnabled", "feeType", "umaResolutionStatus")}, 600))
        m = e["markets"][3] if len(e["markets"]) > 3 else e["markets"][0]
        print("  description:", short(m.get("description"), 1200))
        ids = m.get("clobTokenIds")
        ids = json.loads(ids) if isinstance(ids, str) else ids
        if ids:
            s, j, h = get(f"{POLY_CLOB}/prices-history", {"market": ids[0], "interval": "max", "fidelity": 60})
            hist = j.get("history", []) if isinstance(j, dict) else []
            print("  prices-history fidelity=60:", s, len(hist), hist[:1], hist[-1:])
            s, j, h = get(f"{POLY_CLOB}/book", {"token_id": ids[0]})
            print("  book (closed market):", s, short(j, 300))
            s, j, h = get("https://data-api.polymarket.com/trades", {"market": m.get("conditionId"), "limit": 50})
            print("  data-api trades:", s, short(j, 400))


def openmeteo_prev():
    section("OPEN-METEO previous-runs: hourly previous_dayN availability by start year (NYC)")
    base = {"latitude": 40.7789, "longitude": -73.9692, "temperature_unit": "fahrenheit", "timezone": "America/New_York"}
    for model in ("gfs_seamless", "ecmwf_ifs025", "icon_seamless", "ecmwf_ifs", "gem_seamless", "gfs_global"):
        for start in ("2021-06-01", "2022-06-01", "2023-06-01", "2024-06-01", "2025-06-01"):
            end = start[:8] + "08"
            s, j, h = get("https://previous-runs-api.open-meteo.com/v1/forecast",
                          {**base, "models": model, "start_date": start, "end_date": end,
                           "hourly": "temperature_2m,temperature_2m_previous_day1,temperature_2m_previous_day2"})
            if s != 200:
                print(model, start, "status", s, short(j, 160))
                continue
            hr = j.get("hourly", {})
            cnt = {k: sum(1 for x in v if x is not None) for k, v in hr.items() if k != "time"}
            print(model, start, "non-null of", len(hr.get("time", [])), cnt)
    s, j, h = get("https://previous-runs-api.open-meteo.com/v1/forecast",
                  {**base, "models": "gfs_seamless", "start_date": "2025-06-01", "end_date": "2025-06-03",
                   "daily": "temperature_2m_max_previous_day1"})
    print("daily previous_day1:", s, short(j, 200))
    section("OPEN-METEO historical-forecast: model start dates (non-null count Jan 2022 / Jan 2023 / Jan 2024)")
    for model in ("ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless", "best_match"):
        out = []
        for start in ("2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"):
            s, j, h = get("https://historical-forecast-api.open-meteo.com/v1/forecast",
                          {**base, "models": model, "start_date": start, "end_date": start[:8] + "28",
                           "daily": "temperature_2m_max"})
            v = (j.get("daily", {}) if isinstance(j, dict) else {})
            vals = [x for k, a in v.items() if k != "time" for x in a if x is not None]
            out.append((start[:4], s, len(vals)))
        print(model, out)


def settle_compare(keep):
    section("SETTLEMENT: Kalshi expiration_value (The Weather Company) vs NWS CLI (IEM) for NYC")
    mk = keep.get("KXHIGHNY") or []
    byday = {}
    for m in mk:
        ev = m.get("event_ticker", "")
        if m.get("expiration_value") not in (None, ""):
            byday[ev] = f(m["expiration_value"])
    s, j, h = get("https://mesonet.agron.iastate.edu/json/cli.py", {"station": "KNYC", "year": 2026})
    cli = {r["valid"]: r["high"] for r in j.get("results", [])} if isinstance(j, dict) else {}
    diffs = []
    mon = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
    for ev, v in byday.items():
        d = ev.split("-")[-1]
        try:
            day = f"20{d[:2]}-{mon[d[2:5]]:02d}-{d[5:]}"
        except (KeyError, ValueError):
            continue
        if day in cli and cli[day] is not None:
            diffs.append(v - cli[day])
    print("days compared:", len(diffs), "distribution of (WeatherCo - CLI):", dict(collections.Counter(diffs)))


if __name__ == "__main__":
    for fn in (kalshi_hist, pm_depth, openmeteo_prev):
        try:
            r = fn()
            if fn is kalshi_hist:
                settle_compare(r)
        except Exception as e:
            print(f"!! {fn.__name__} crashed: {type(e).__name__}: {e}")
