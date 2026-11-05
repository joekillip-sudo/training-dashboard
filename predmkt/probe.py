"""STEP 0 feasibility probe. Read-only, polite. Prints what exists; builds nothing.

Run on a GitHub Actions runner (the dev sandbox cannot reach these sites):
    python -m predmkt.probe > probe_report.txt
"""
import collections
import json
import sys
import time

from .http import get

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"


def short(x, n=700):
    s = x if isinstance(x, str) else json.dumps(x)
    return s[:n] + ("..." if len(s) > n else "")


def hdrs(h):
    return {k: v for k, v in h.items() if any(t in k.lower() for t in ("rate", "limit", "retry", "remaining"))}


def section(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}", flush=True)


def kalshi():
    section("KALSHI: series list (weather)")
    st, j, h = get(f"{KALSHI}/series", {"category": "Climate and Weather"})
    print("status", st, "ratelimit-ish headers", hdrs(h))
    series = j.get("series", []) if isinstance(j, dict) else []
    print("series returned:", len(series))
    temp = [s for s in series if any(w in (s.get("title", "") + s.get("ticker", "")).lower()
                                     for w in ("temperature", "high", "low", "kxhigh", "kxlow"))]
    for s in temp[:60]:
        print(" ", s.get("ticker"), "|", s.get("title"), "|", s.get("frequency"))
    if not series:
        print(short(j))
    tickers = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHLAX", "KXHIGHDEN", "KXHIGHPHIL",
               "HIGHNY", "HIGHCHI"]
    sample = None
    for sr in tickers:
        section(f"KALSHI: settled markets for {sr}")
        mk, cursor, pages = [], None, 0
        while pages < 30:
            p = {"series_ticker": sr, "status": "settled", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            st, j, h = get(f"{KALSHI}/markets", p)
            if st != 200 or not isinstance(j, dict):
                print("status", st, short(j))
                break
            mk += j.get("markets", [])
            cursor = j.get("cursor")
            pages += 1
            if not cursor:
                break
        print("settled markets:", len(mk), "pages:", pages)
        if not mk:
            continue
        by_month = collections.Counter(m.get("close_time", "")[:7] for m in mk)
        print("by month:", dict(sorted(by_month.items())))
        print("events:", len({m.get("event_ticker") for m in mk}))
        print("example market keys:", sorted(mk[0].keys()))
        print("example:", short({k: mk[0].get(k) for k in (
            "ticker", "event_ticker", "title", "subtitle", "yes_sub_title", "status", "result", "open_time",
            "close_time", "expiration_time", "floor_strike", "cap_strike", "strike_type", "volume", "open_interest",
            "liquidity", "last_price", "previous_price", "yes_bid", "yes_ask", "settlement_value",
            "rules_primary", "rules_secondary")}, 1800))
        if sample is None:
            sample = (sr, sorted(mk, key=lambda m: m.get("close_time", ""))[len(mk) // 2])
        vols = sorted(m.get("volume", 0) or 0 for m in mk)
        print("volume per market: median", vols[len(vols) // 2], "p90", vols[int(len(vols) * .9)], "max", vols[-1])
    if sample:
        sr, m = sample
        section(f"KALSHI: price history & trades for {m['ticker']}")
        ct = m.get("close_time", "")
        ot = m.get("open_time", "")
        print("open", ot, "close", ct)
        import datetime as dt

        def ts(s):
            return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
        try:
            for per in (60, 1440):
                st, j, h = get(f"{KALSHI}/series/{sr}/markets/{m['ticker']}/candlesticks",
                               {"start_ts": ts(ot), "end_ts": ts(ct), "period_interval": per})
                c = j.get("candlesticks", []) if isinstance(j, dict) else []
                print(f"candlesticks period={per}min status {st} n={len(c)}")
                if c:
                    print("  first:", short(c[0], 900))
                    print("  last:", short(c[-1], 900))
                else:
                    print(" ", short(j))
        except Exception as e:
            print("candlestick error", e)
        st, j, h = get(f"{KALSHI}/markets/trades", {"ticker": m["ticker"], "limit": 1000})
        t = j.get("trades", []) if isinstance(j, dict) else []
        print("trades status", st, "n", len(t), "cursor", (j.get("cursor") if isinstance(j, dict) else None))
        if t:
            print("  first:", short(t[0]), "\n  last:", short(t[-1]))
        else:
            print(short(j))
        st, j, h = get(f"{KALSHI}/markets/{m['ticker']}/orderbook")
        print("orderbook (live only, settled market) status", st, short(j, 300))
    section("KALSHI: historical cutoff / archive endpoints")
    for path in ("/historical/cutoff",):
        st, j, h = get(KALSHI + path)
        print(path, st, short(j))
    st, j, h = get(f"{KALSHI}/historical/markets", {"series_ticker": "KXHIGHNY", "limit": 5})
    print("/historical/markets", st, short(j, 900))
    section("KALSHI: exchange fee info")
    st, j, h = get(f"{KALSHI}/series/KXHIGHNY")
    s = j.get("series", {}) if isinstance(j, dict) else {}
    print("series KXHIGHNY:", short({k: s.get(k) for k in ("ticker", "title", "fee_type", "fee_multiplier", "frequency",
          "settlement_sources", "contract_url", "contract_terms_url", "tags", "category")}, 1500))


def polymarket():
    section("POLYMARKET: find temperature markets")
    found = []
    for q in ("highest temperature", "temperature in"):
        st, j, h = get(f"{POLY_GAMMA}/public-search", {"q": q, "limit_per_type": 20, "events_status": "closed"})
        print(f"search '{q}' status {st} hdr {hdrs(h)}")
        evs = j.get("events", []) if isinstance(j, dict) else []
        print(" events:", len(evs))
        for e in evs[:20]:
            print("  ", e.get("slug"), "|", e.get("title"), "|", e.get("endDate"), "| markets", len(e.get("markets", [])))
            found.append(e)
        if not evs:
            print(short(j, 400))
    st, j, h = get(f"{POLY_GAMMA}/events", {"closed": "true", "limit": 100, "order": "endDate", "ascending": "false",
                                            "tag_slug": "weather"})
    evs = j if isinstance(j, list) else []
    print("events by tag weather (closed):", st, len(evs))
    for e in evs[:15]:
        print("  ", e.get("slug"), "|", e.get("title"), "|", e.get("endDate"), "| vol", e.get("volume"))
    found += evs
    tok = None
    for e in found:
        for m in e.get("markets", []):
            ids = m.get("clobTokenIds")
            if isinstance(ids, str):
                try:
                    ids = json.loads(ids)
                except ValueError:
                    ids = None
            if ids and "temperature" in (m.get("question", "") + e.get("title", "")).lower():
                tok = (m, ids[0])
                break
        if tok:
            break
    if tok:
        m, t = tok
        print("\nexample market:", short({k: m.get(k) for k in ("question", "conditionId", "startDate", "endDate",
              "volume", "liquidity", "outcomes", "outcomePrices", "resolutionSource", "description")}, 1800))
        for fid in (60, 1440):
            st, j, h = get(f"{POLY_CLOB}/prices-history", {"market": t, "interval": "max", "fidelity": fid})
            hist = j.get("history", []) if isinstance(j, dict) else []
            print(f"prices-history fidelity={fid} status {st} n={len(hist)} hdr {hdrs(h)}")
            if hist:
                print("  first", hist[0], "last", hist[-1])
            else:
                print(" ", short(j, 300))
    else:
        print("no example temperature market with token ids found")


def openmeteo():
    section("OPEN-METEO: archived forecasts")
    lat, lon = 40.7789, -73.9692  # Central Park
    base = {"latitude": lat, "longitude": lon, "daily": "temperature_2m_max", "temperature_unit": "fahrenheit",
            "timezone": "America/New_York"}
    for name, url, extra in (
        ("historical-forecast 2022-01", "https://historical-forecast-api.open-meteo.com/v1/forecast",
         {"start_date": "2022-01-01", "end_date": "2022-01-10"}),
        ("historical-forecast 2024-06", "https://historical-forecast-api.open-meteo.com/v1/forecast",
         {"start_date": "2024-06-01", "end_date": "2024-06-10"}),
        ("historical-forecast models ecmwf_ifs025,gfs_seamless,icon_seamless",
         "https://historical-forecast-api.open-meteo.com/v1/forecast",
         {"start_date": "2025-06-01", "end_date": "2025-06-05",
          "models": "ecmwf_ifs025,gfs_seamless,icon_seamless"}),
    ):
        st, j, h = get(url, {**base, **extra})
        print(name, "status", st, "hdr", hdrs(h))
        print("  ", short(j, 500))
    # previous-runs: true "as issued N days before"
    for name, extra in (("previous-runs day0..3 gfs", {"models": "gfs_seamless",
                         "daily": "temperature_2m_max,temperature_2m_max_previous_day1,temperature_2m_max_previous_day2",
                         "start_date": "2025-06-01", "end_date": "2025-06-05"}),
                        ("previous-runs early 2024", {"models": "gfs_seamless",
                         "daily": "temperature_2m_max,temperature_2m_max_previous_day1",
                         "start_date": "2024-01-01", "end_date": "2024-01-05"}),
                        ("previous-runs early 2023", {"models": "gfs_seamless",
                         "daily": "temperature_2m_max,temperature_2m_max_previous_day1",
                         "start_date": "2023-01-01", "end_date": "2023-01-05"})):
        st, j, h = get("https://previous-runs-api.open-meteo.com/v1/forecast", {**base, **{k: v for k, v in extra.items()}})
        print(name, "status", st)
        print("  ", short(j, 600))
    st, j, h = get("https://ensemble-api.open-meteo.com/v1/ensemble",
                   {**base, "models": "gfs025", "start_date": "2025-06-01", "end_date": "2025-06-02"})
    print("ensemble-api (live) status", st, short(j, 300))
    st, j, h = get("https://historical-forecast-api.open-meteo.com/v1/forecast",
                   {**base, "models": "gfs025_ensemble" , "start_date": "2025-06-01", "end_date": "2025-06-02"})
    print("historical ensemble try", st, short(j, 300))


def noaa():
    section("SETTLEMENT DATA: NCEI / NWS CLI / IEM")
    st, j, h = get("https://www.ncei.noaa.gov/access/services/data/v1",
                   {"dataset": "daily-summaries", "stations": "USW00094728", "startDate": "2025-06-01",
                    "endDate": "2025-06-05", "dataTypes": "TMAX,TMIN", "units": "standard", "format": "json"})
    print("NCEI daily-summaries NYC Central Park", st, short(j, 500))
    st, j, h = get("https://mesonet.agron.iastate.edu/json/cli.py", {"station": "KNYC", "year": 2025})
    print("IEM CLI json KNYC 2025", st, short(j, 500))
    if isinstance(j, dict):
        print("  rows:", len(j.get("results", [])))
    st, j, h = get("https://api.weather.gov/products/types/CLI/locations/NYC")
    print("NWS products CLI/NYC", st, short(j, 300))


if __name__ == "__main__":
    which = sys.argv[1:] or ["kalshi", "polymarket", "openmeteo", "noaa"]
    t0 = time.time()
    for w in which:
        try:
            globals()[w]()
        except Exception as e:  # keep going: a failed probe is itself a finding
            print(f"!! {w} crashed: {type(e).__name__}: {e}")
    print(f"\ndone in {time.time() - t0:.0f}s")
