"""Load football-data.co.uk CSVs, and generate synthetic data for tests."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
# opening odds (Bet365) and sharp closing odds (Pinnacle) when present
ODDS_COLS = ["B365H", "B365D", "B365A", "PSCH", "PSCD", "PSCA"]


def fetch(leagues, seasons, out_dir="picks/data"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        for lg in leagues:
            df = pd.read_csv(BASE.format(season=s, league=lg), encoding="latin-1")
            df.to_csv(out / f"{lg}_{s}.csv", index=False)
            print(f"{lg} {s}: {len(df)} matches")


def load(data_dir="picks/data"):
    frames = []
    for f in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(f)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
        df["League"] = f.stem.split("_")[0]
        cols = [c for c in KEEP + ODDS_COLS if c in df.columns]
        frames.append(df[cols + ["League"]])
    return pd.concat(frames, ignore_index=True).dropna(subset=["FTHG"]).sort_values("Date").reset_index(drop=True)


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
            rows.append({
                "Date": day + pd.Timedelta(days=i // 10), "HomeTeam": teams[h], "AwayTeam": teams[a],
                "FTHG": hg, "FTAG": ag, "FTR": "HDA"[int(np.argmax([hg > ag, hg == ag, hg < ag]))],
                "B365H": odds[0], "B365D": odds[1], "B365A": odds[2],
                # closing line: much closer to the truth
                "PSCH": 1 / (p[0] * 1.02), "PSCD": 1 / (p[1] * 1.02), "PSCA": 1 / (p[2] * 1.02),
                "League": "SYN",
            })
        day += pd.Timedelta(days=i // 10 + 60)
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
