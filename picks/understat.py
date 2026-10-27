"""Scrape real xG from Understat and attach it to football-data results.

Understat has no official API. Each league-season page embeds every match as JSON in a
`var datesData = JSON.parse('...')` script with \\xHH escapes; we parse that. It is the part
most likely to break if the site changes, so the parser raises rather than guessing.

Scrape politely: one page per league-season, a pause between requests, cached to CSV.
"""
import difflib
import gzip
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

LEAGUES = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1"}
URL = "https://understat.com/league/{league}/{year}"
API_URL = "https://understat.com/getLeagueData/{league}/{year}"  # JSON the page itself loads by script
USER_AGENT = "Mozilla/5.0 (compatible; football-picks-research/1.0)"
FUZZY_CUTOFF = 0.85
DATE_TOLERANCE = pd.Timedelta(days=3)  # kickoff timezone / postponement slack

# Understat name -> football-data name. Anything missing falls back to fuzzy matching, and
# anything still unmatched is reported. Extend or override via picks/team_aliases.json.
ALIASES = {
    "Manchester United": "Man United", "Manchester City": "Man City", "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves", "Nottingham Forest": "Nott'm Forest", "West Bromwich Albion": "West Brom",
    "Sheffield United": "Sheffield United", "Leeds": "Leeds", "Tottenham": "Tottenham",
    "Atletico Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao", "Celta Vigo": "Celta",
    "Real Betis": "Betis", "Real Sociedad": "Sociedad", "Rayo Vallecano": "Vallecano", "Espanyol": "Espanol",
    "Borussia M.Gladbach": "M'gladbach", "Borussia Dortmund": "Dortmund", "Bayer Leverkusen": "Leverkusen",
    "RasenBallsport Leipzig": "RB Leipzig", "Eintracht Frankfurt": "Ein Frankfurt", "FC Cologne": "FC Koln",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf", "Mainz 05": "Mainz", "VfB Stuttgart": "Stuttgart",
    "AC Milan": "Milan", "Parma Calcio 1913": "Parma", "Paris Saint Germain": "Paris SG",
    "Saint-Etienne": "St Etienne", "Olympique Marseille": "Marseille", "Olympique Lyonnais": "Lyon",
}

_DATES = re.compile(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", re.S)


def _rows(games):
    rows = []
    for g in games:
        if not g.get("isResult"):
            continue
        rows.append({
            "Date": pd.Timestamp(g["datetime"]).normalize(),
            "HomeTeam": g["h"]["title"], "AwayTeam": g["a"]["title"],
            "HxG": float(g["xG"]["h"]), "AxG": float(g["xG"]["a"]),
        })
    return pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "HxG", "AxG"])


def parse_matches(html):
    """Parse a league page into played matches with Understat team names and xG."""
    found = _DATES.search(html)
    if not found:
        raise ValueError("datesData not found: Understat's page layout may have changed")
    text = re.sub(r"\\x([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), found.group(1))
    return _rows(json.loads(text))


def parse_league_json(text):
    """Parse the getLeagueData JSON ({"teams":…, "players":…, "dates":[matches]})."""
    return _rows(json.loads(text)["dates"])


def fetch(leagues=tuple(LEAGUES), seasons=("2526",), out_dir="picks/xg_data", delay=2.0, opener=None):
    """Download xG for football-data season codes (e.g. '2526' -> Understat year 2025)."""
    opener = opener or (lambda req: urllib.request.urlopen(req, timeout=30))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for s in seasons:
        for lg in leagues:
            if lg not in LEAGUES:
                continue
            year = 2000 + int(s[:2])
            page = URL.format(league=LEAGUES[lg], year=year)
            attempts = (
                (API_URL.format(league=LEAGUES[lg], year=year), parse_league_json,
                 {"X-Requested-With": "XMLHttpRequest", "Referer": page}),
                (page, parse_matches, {}),
            )
            df, problems = None, []
            for url, parse, extra in attempts:
                try:
                    raw = opener(urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **extra})).read()
                    if raw[:2] == b"\x1f\x8b":  # Understat gzips its responses regardless of Accept-Encoding
                        raw = gzip.decompress(raw)
                    body = raw.decode("utf-8", "replace")
                except Exception as e:
                    problems.append(f"{url.rsplit('/', 3)[-3]} request failed: {e}")
                    continue
                try:
                    df = parse(body)
                    break
                except Exception as e:  # xG is optional; never sink the whole refresh
                    problems.append(f"{url.rsplit('/', 3)[-3]} unparseable ({e}); {len(body)} bytes starting {body[:80]!r}")
            if df is None:
                print(f"WARN understat {lg} {s}: " + " | ".join(problems))
                continue
            df.to_csv(out / f"{lg}_{s}.csv", index=False)
            ok += 1
            print(f"understat {lg} {s}: {len(df)} matches")
            time.sleep(delay)
    return ok


def _norm(name):
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(fc|afc|cf|calcio|sc|ac)\b", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


def map_teams(us_names, fd_names, aliases=None):
    """Map Understat names to football-data names. Returns (mapping, unmatched).

    Ambiguous results (two Understat teams landing on one football-data team) are rejected
    rather than guessed, since a wrong join silently corrupts ratings.
    """
    aliases = {**ALIASES, **(aliases or {})}
    fd = {_norm(n): n for n in fd_names}
    mapping = {}
    for name in sorted(us_names):
        target = aliases.get(name)
        if target in fd_names:
            mapping[name] = target
        elif _norm(name) in fd:
            mapping[name] = fd[_norm(name)]
        else:
            close = difflib.get_close_matches(_norm(name), list(fd), n=1, cutoff=FUZZY_CUTOFF)
            if close:
                mapping[name] = fd[close[0]]
    counts = pd.Series(list(mapping.values())).value_counts()
    for name in [n for n, t in mapping.items() if counts[t] > 1]:
        del mapping[name]
    return mapping, sorted(set(us_names) - set(mapping))


def attach_xg(df, xg_dir="picks/xg_data", aliases_path="picks/team_aliases.json"):
    """Add HxG/AxG columns to football-data results where Understat has the match."""
    files = sorted(Path(xg_dir).glob("*.csv")) if Path(xg_dir).is_dir() else []
    if not files:
        return df
    aliases = json.loads(Path(aliases_path).read_text()) if Path(aliases_path).exists() else {}
    xg = pd.concat([pd.read_csv(f, parse_dates=["Date"]).assign(League=f.stem.split("_")[0]) for f in files])
    out = df.copy()
    out["HxG"] = out["AxG"] = float("nan")
    key = out[["League", "HomeTeam", "AwayTeam", "Date"]].reset_index()
    for lg, g in xg.groupby("League"):
        names = set(out.loc[out["League"] == lg, "HomeTeam"]) | set(out.loc[out["League"] == lg, "AwayTeam"])
        mapping, unmatched = map_teams(set(g["HomeTeam"]) | set(g["AwayTeam"]), names, aliases)
        if unmatched:
            print(f"WARN {lg}: no football-data match for {unmatched}; add them to {aliases_path}")
        g = g.assign(HomeTeam=g["HomeTeam"].map(mapping), AwayTeam=g["AwayTeam"].map(mapping)).dropna(subset=["HomeTeam", "AwayTeam"])
        m = key[key["League"] == lg].merge(g, on=["League", "HomeTeam", "AwayTeam"], suffixes=("", "_x"))
        m = m[(m["Date"] - m["Date_x"]).abs() <= DATE_TOLERANCE].drop_duplicates("index")
        out.loc[m["index"], ["HxG", "AxG"]] = m[["HxG", "AxG"]].to_numpy()
    return out
