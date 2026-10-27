"""Load football-data.co.uk CSVs, and generate synthetic data for tests."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .understat import attach_xg

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
# opening odds (Bet365) and sharp closing odds (Pinnacle) when present
SHOT_COLS = ["HS", "AS", "HST", "AST"]
ODDS_COLS = ["B365H", "B365D", "B365A", "PSCH", "PSCD", "PSCA",
             "PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
             "AvgCH", "AvgCD", "AvgCA"]  # pre-match: Pinnacle, best, average


def season_codes(n=4, today=None):
    """Last n season codes ending with the current one, e.g. ['2223', '2324', '2425', '2526']."""
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(start - n + 1, start + 1)]


def fetch(leagues, seasons, out_dir="picks/data"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for s in seasons:
        for lg in leagues:
            try:
                df = pd.read_csv(BASE.format(season=s, league=lg), encoding="latin-1")
            except Exception as e:  # a missing file shouldn't sink the whole refresh
                print(f"WARN {lg} {s}: {e}")
                continue
            df.to_csv(out / f"{lg}_{s}.csv", index=False)
            ok += 1
            print(f"{lg} {s}: {len(df)} matches")
    if not ok:
        raise SystemExit("no data downloaded")


def fetch_fixtures(path="picks/fixtures.csv"):
    df = pd.read_csv(FIXTURES_URL, encoding="latin-1")
    df.to_csv(path, index=False)
    print(f"fixtures: {len(df)} rows -> {path}")


def load_fixtures(path="picks/fixtures.csv", leagues=None, from_date=None):
    """Upcoming fixtures. The league column is optional; see infer_leagues()."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = {"Date", "HomeTeam", "AwayTeam"} - set(df.columns)
    if missing:
        raise ValueError(f"fixtures file lacks {sorted(missing)}; columns are {list(df.columns)}")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    by_lower = {c.lower(): c for c in df.columns}
    for alias in ("div", "league", "division"):
        if alias in by_lower:
            df = df.rename(columns={by_lower[alias]: "League"})
            break
    if leagues is not None and "League" in df.columns:
        df = df[df["League"].isin(leagues)]
    if from_date is not None:
        df = df[df["Date"] >= pd.Timestamp(from_date)]
    return df.reset_index(drop=True)


def infer_leagues(fixtures, models):
    """Add League to fixtures that lack it: the one league whose history has both teams.

    Rows matching no league, or several (e.g. both teams moved division), are dropped.
    """
    def pick(row):
        hits = [lg for lg, m in models.items() if m.knows(row["HomeTeam"]) and m.knows(row["AwayTeam"])]
        return hits[0] if len(hits) == 1 else None
    out = fixtures.copy()
    out["League"] = out.apply(pick, axis=1)
    return out.dropna(subset=["League"]).reset_index(drop=True)


def load(data_dir="picks/data"):
    frames = []
    for f in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(f).dropna(subset=["Date", "HomeTeam"])
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
        cols = [c for c in KEEP + SHOT_COLS + ODDS_COLS if c in df.columns]
        frames.append(df[cols].assign(League=f.stem.split("_")[0]))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["FTHG"]).sort_values("Date").reset_index(drop=True)
    return attach_xg(df, Path(data_dir).parent / "xg_data")


def _norm3(v):
    return v / v.sum()


def synthetic(n_teams=20, seasons=3, margin=0.05, noise=0.10, seed=0):
    """Round-robin league with known true strengths and noisy bookmaker odds.

    Bookmaker prices = true probabilities perturbed by lognormal noise, plus margin.
    Useful for checking the pipeline end to end, not for judging real-world profit.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    att = np.exp(rng.normal(0, 0.25, n_teams))
    dfn = np.exp(rng.normal(0, 0.25, n_teams))
    rows, day = [], pd.Timestamp("2022-08-01")
    for _ in range(seasons):
        fixtures = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(fixtures)
        for i, (h, a) in enumerate(fixtures):
            lh, la = att[h] * dfn[a] * 1.45, att[a] * dfn[h] * 1.1
            hg, ag = rng.poisson(lh), rng.poisson(la)
            g = np.arange(11)
            from math import exp, factorial
            ph = np.array([exp(-lh) * lh**k / factorial(k) for k in g])
            pa = np.array([exp(-la) * la**k / factorial(k) for k in g])
            m = np.outer(ph, pa)
            p = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
            p /= p.sum()
            q = p * np.exp(rng.normal(0, noise, 3))
            q /= q.sum()
            odds = 1 / (q * (1 + margin))
            hs, as_ = rng.poisson(lh * 9), rng.poisson(la * 9)  # shots track true chance quality
            rows.append({
                "HS": hs, "AS": as_, "HST": rng.binomial(hs, 0.33), "AST": rng.binomial(as_, 0.33),
                "Date": day + pd.Timedelta(days=i * 280 // len(fixtures)), "HomeTeam": teams[h], "AwayTeam": teams[a],
                "FTHG": hg, "FTAG": ag, "FTR": "HDA"[int(np.argmax([hg > ag, hg == ag, hg < ag]))],
                "B365H": odds[0], "B365D": odds[1], "B365A": odds[2],
                **{f"{k}{o}": v for k, vals in (("PS", 1 / (_norm3(p * np.exp(rng.normal(0, 0.03, 3))) * 1.02)),
                                                 ("Avg", 1 / (_norm3(0.5 * q + 0.5 * p) * 1.04)),
                                                 ("Max", odds * (1 + np.abs(rng.normal(0, 0.04, 3)))))
                   for o, v in zip("HDA", vals)},
                # closing line: much closer to the truth
                "PSCH": 1 / (p[0] * 1.02), "PSCD": 1 / (p[1] * 1.02), "PSCA": 1 / (p[2] * 1.02),
                "League": "SYN",
            })
        day += pd.Timedelta(days=365)
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
