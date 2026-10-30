"""Generated tennis-data-shaped files with a known truth, to test the machinery and as controls.

noise_sd = 0: bookmaker prices are the true probabilities plus margin (an efficient market; nothing to find).
noise_sd > 0: prices are the truth blurred by random error, so a model that learns skill from results CAN win.
"""
import numpy as np
import pandas as pd

LEVELS = ["Grand Slam", "Masters 1000", "ATP500", "ATP250", "ATP250", "ATP250"]
ROUNDS = ["1st Round", "1st Round", "2nd Round", "2nd Round", "3rd Round", "Quarterfinals", "Semifinals", "The Final"]


def _odds_from(p, margin):
    """Two decimal prices for probabilities (p, 1-p) carrying the given overround."""
    return 1 / (p * (1 + margin)), 1 / ((1 - p) * (1 + margin))


def synth_raw(n_matches=24000, n_players=160, start="2005-01-01", years=14, noise_sd=0.0, seed=1, retire_rate=0.03):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, n_players)
    surf_fx = {s: rng.normal(0, 0.35, n_players) for s in ("Hard", "Clay", "Grass")}
    names = [f"Player{i:03d} X." for i in range(n_players)]
    order = np.argsort(-skill)
    rank_of = np.empty(n_players, int)
    rank_of[order] = np.arange(1, n_players + 1)
    pts = 10000 * np.exp(-0.03 * rank_of)
    days = np.sort(rng.integers(0, int(365.25 * years), n_matches))
    dates = pd.Timestamp(start) + pd.to_timedelta(days, unit="D")
    rows = []
    for k in range(n_matches):
        a, b = rng.choice(n_players, 2, replace=False)
        surf = rng.choice(["Hard", "Hard", "Clay", "Grass"])
        z = 1.1 * (skill[a] + surf_fx[surf][a] - skill[b] - surf_fx[surf][b])
        p_true = 1 / (1 + np.exp(-z))
        a_wins = rng.random() < p_true
        w, l = (a, b) if a_wins else (b, a)
        pm = np.clip(1 / (1 + np.exp(-(z + rng.normal(0, noise_sd)))) if noise_sd else p_true, 0.01, 0.99)  # price's view of player a
        paw = pm if a_wins else 1 - pm
        ow, ol = _odds_from(paw, 0.03)
        aw, al = _odds_from(paw, 0.06)
        mw, ml = aw * 1.03, al * 1.03
        sets = 2 if rng.random() < 0.95 else 1
        rows.append({
            "Tournament": f"Open {dates[k]:%Y%W}", "Date": dates[k], "Series": rng.choice(LEVELS), "Surface": surf,
            "Round": rng.choice(ROUNDS), "Best of": 3, "Winner": names[w], "Loser": names[l],
            "WRank": rank_of[w], "LRank": rank_of[l], "WPts": pts[w], "LPts": pts[l],
            "Wsets": sets, "Lsets": 0 if sets == 2 else 1,
            "Comment": "Retired" if rng.random() < retire_rate else "Completed",
            "PSW": ow, "PSL": ol, "AvgW": aw, "AvgL": al, "MaxW": mw, "MaxL": ml, "B365W": aw * 0.98, "B365L": al * 0.98,
        })
    return pd.DataFrame(rows)
