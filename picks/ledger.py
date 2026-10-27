"""Paper-trading ledger: every candidate bet is frozen when first seen, then settled later.

The ledger stores candidates (model and market probabilities plus the odds on offer) rather
than sized bets, so the dashboard can re-run the whole simulation under any staking rule or
threshold the viewer chooses. Entries are never rewritten after being placed, which keeps the
live record free of hindsight. Settings can only be *tightened* below the stored floor.
"""
import json
from pathlib import Path

import pandas as pd

from .backtest import CLOSE, OPEN, walk_forward
from .value import devig, find_picks

FLOOR = {"min_ev": 0.02, "min_edge": 0.01, "max_odds": 15.0}
KICKOFF_BUFFER = pd.Timedelta(minutes=30)  # a bet must be placed at least this long before kickoff
HORIZON_DAYS = 2  # only bet fixtures kicking off within this many days (odds are closer to final)
DATE_TOLERANCE = pd.Timedelta(days=3)


def bet_id(league, date, home, away, outcome):
    return f"{league}|{date}|{home}|{away}|{outcome}"


def load(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {"floor": FLOOR, "live_from": None, "generated": None, "bets": []}


def save(book, path):
    """One bet per line so git diffs stay readable."""
    bets = sorted(book["bets"], key=lambda b: (b["date"], b["id"]))
    head = json.dumps({k: v for k, v in book.items() if k != "bets"}, separators=(",", ":"))[:-1]
    body = ",\n".join(json.dumps(b, separators=(",", ":")) for b in bets)
    Path(path).write_text(f'{head},"bets":[\n{body}\n]}}\n')


AVG_CLOSE = {"H": "AvgCH", "D": "AvgCD", "A": "AvgCA"}  # stands in when Pinnacle's close is missing


def _close_p(row, outcome):
    for cols in (CLOSE, AVG_CLOSE):
        if all(c in row and not pd.isna(row[c]) and row[c] > 1 for c in cols.values()):
            return round(devig({o: row[c] for o, c in cols.items()})[outcome], 4)
    return None


def _entries(match_row, home, away, date, model_probs, odds, source, result=None, close_row=None):
    out = []
    for p in find_picks(model_probs, odds, **FLOOR):
        out.append({
            "id": bet_id(match_row["League"], date, home, away, p.outcome),
            "league": match_row["League"], "date": date, "home": home, "away": away,
            "outcome": p.outcome, "odds": round(p.odds, 3),
            "model_p": round(p.model_p, 4), "market_p": round(p.market_p, 4),
            "source": source, "result": result,
            "close_p": _close_p(close_row, p.outcome) if close_row is not None else None,
        })
    return out


def _bettable(m, now):
    """True if the match has not started. Uses the fixture's UK kickoff time when given; without a
    time only matches on a later day qualify, since the match could already be over."""
    time = m.get("Time")
    if isinstance(time, str) and time.strip():
        try:
            kickoff = pd.Timestamp(f"{m['Date'].date()} {time.strip()}", tz="Europe/London").tz_convert("UTC")
            return kickoff > pd.Timestamp(now) + KICKOFF_BUFFER
        except (ValueError, TypeError):
            pass
    return m["Date"].date() > now.date()


def add_live(book, fixtures, models, now, horizon=HORIZON_DAYS):
    """Record value bets on upcoming fixtures. Existing entries are never touched.

    `now` is a timezone-aware UTC datetime: matches already started (or whose start can't be
    ruled out) are skipped so the live record contains only bets that could really be placed.
    """
    if book.get("live_from") is None:
        book["live_from"] = now.date().isoformat()
    have = {b["id"] for b in book["bets"]}
    last = pd.Timestamp(now.date()) + pd.Timedelta(days=horizon)
    added = 0
    for _, m in fixtures.iterrows():
        if m["Date"] > last or not _bettable(m, now):
            continue
        model = models.get(m["League"])
        odds = {"H": m.get("B365H"), "D": m.get("B365D"), "A": m.get("B365A")}
        if model is None or any(pd.isna(v) for v in odds.values()):
            continue
        if not (model.knows(m["HomeTeam"]) and model.knows(m["AwayTeam"])):
            continue
        date = str(m["Date"].date())
        for e in _entries(m, m["HomeTeam"], m["AwayTeam"], date, model.predict(m["HomeTeam"], m["AwayTeam"]), odds, "live"):
            if e["id"] not in have:
                book["bets"].append(e)
                have.add(e["id"])
                added += 1
    return added


def seed_backtest(book, df, before=None, **model_kw):
    """One-off: add walk-forward historical candidates (source='backtest') dated before `before`."""
    if any(b["source"] == "backtest" for b in book["bets"]):
        return 0
    before = pd.Timestamp(before or book.get("live_from") or pd.Timestamp.today().date())
    added = 0
    for m, model in walk_forward(df, **model_kw):
        if m["Date"] >= before or any(pd.isna(m.get(c)) for c in OPEN.values()):
            continue
        odds = {o: m[c] for o, c in OPEN.items()}
        date = str(m["Date"].date())
        es = _entries(m, m["HomeTeam"], m["AwayTeam"], date, model.predict(m["HomeTeam"], m["AwayTeam"]),
                      odds, "backtest", result=m["FTR"], close_row=m)
        book["bets"].extend(es)
        added += len(es)
    return added


def settle(book, df):
    """Fill in results (and closing price, for CLV) for open bets once the match is in `df`."""
    groups = {k: g for k, g in df.groupby(["League", "HomeTeam", "AwayTeam"])}
    settled = 0
    for b in book["bets"]:
        if b["result"] is not None:
            continue
        g = groups.get((b["league"], b["home"], b["away"]))
        if g is None:
            continue
        gap = (g["Date"] - pd.Timestamp(b["date"])).abs()
        if gap.min() > DATE_TOLERANCE:
            continue
        row = g.loc[gap.idxmin()]
        b["result"] = row["FTR"]
        b["close_p"] = _close_p(row, b["outcome"])
        settled += 1
    return settled
