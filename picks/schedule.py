"""Does rest and fixture congestion explain anything the model (or the market) misses?

Features come only from the match dates already in the data: days since a team's previous league
match, and how many league matches it played in the previous 14 days. Cup and European games are
not in the data, so congestion at big clubs is understated; that is a real limit of this test.

The correction is a small regularised multinomial fit on top of a base forecast, refitted monthly on
earlier matches only, so the results are out of sample.
"""
import numpy as np
import pandas as pd

from .market import ORDER, Y

REST_CAP = (2, 14)   # days; also stops the summer break counting as "rested"
FILL_REST = 7
NAMES = ("rest advantage", "congestion (matches in 14d)", "short rest (<=3d)")


def add_schedule_features(df):
    """Return df sorted by date with rest_h/rest_a (days since own previous league match) and
    games14_h/games14_a (own league matches in the previous 14 days)."""
    d = df.sort_values("Date", kind="stable").reset_index(drop=True)
    long = pd.concat([
        pd.DataFrame({"row": d.index, "side": "h", "team": d["League"] + "|" + d["HomeTeam"], "Date": d["Date"]}),
        pd.DataFrame({"row": d.index, "side": "a", "team": d["League"] + "|" + d["AwayTeam"], "Date": d["Date"]}),
    ]).sort_values(["team", "Date", "row"], kind="stable")
    rest, games = np.full(len(long), np.nan), np.zeros(len(long))
    pos = 0
    for _, g in long.groupby("team", sort=False):
        dates = g["Date"].to_numpy("datetime64[ns]")
        n = len(dates)
        if n > 1:
            rest[pos + 1:pos + n] = (dates[1:] - dates[:-1]) / np.timedelta64(1, "D")
        first = np.searchsorted(dates, dates - np.timedelta64(14, "D"), side="left")
        games[pos:pos + n] = np.arange(n) - first
        pos += n
    long["rest"], long["games14"] = rest, games
    for side in ("h", "a"):
        part = long[long["side"] == side].set_index("row")
        d[f"rest_{side}"] = part["rest"].reindex(d.index).to_numpy()
        d[f"games14_{side}"] = part["games14"].reindex(d.index).to_numpy()
    return d


def feature_tensor(t):
    """n x 3 outcomes x 3 features. Positive weight = the home side benefits; away gets the mirror."""
    lo, hi = REST_CAP
    rh = np.clip(t["rest_h"].fillna(FILL_REST).to_numpy(float), lo, hi)
    ra = np.clip(t["rest_a"].fillna(FILL_REST).to_numpy(float), lo, hi)
    X = np.column_stack([
        (rh - ra) / 5.0,
        -(t["games14_h"].to_numpy(float) - t["games14_a"].to_numpy(float)) / 2.0,  # fewer recent games is better
        ((rh <= 3).astype(float) * -1) - ((ra <= 3).astype(float) * -1),            # short rest hurts
    ])
    F = np.zeros((len(t), 3, X.shape[1]))
    F[:, 0, :], F[:, 2, :] = X, -X
    return F


def fit(F, base_p, y, lam=20.0, iters=30):
    k = F.shape[2]
    off, Yoh, theta = np.log(np.clip(base_p, 1e-9, 1)), np.eye(3)[y], np.zeros(k)
    for _ in range(iters):
        s = off + F @ theta
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(1, keepdims=True)
        Ef = (p[:, :, None] * F).sum(1)
        g = (Ef - (Yoh[:, :, None] * F).sum(1)).sum(0) + 2 * lam * theta
        Eff = (p[:, :, None, None] * F[:, :, :, None] * F[:, :, None, :]).sum(1)
        H = (Eff - Ef[:, :, None] * Ef[:, None, :]).sum(0) + 2 * lam * np.eye(k)
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-7:
            break
    return theta


def predict(theta, F, base_p):
    s = np.log(np.clip(base_p, 1e-9, 1)) + F @ theta
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(1, keepdims=True)


def oos(tbl, base, min_train=1500, step_days=30, lam=20.0):
    """Out-of-sample adjusted probabilities for one base ('model' or 'avg'). Returns (rows, last theta)."""
    cols = [f"model_{o}" if base == "model" else f"{base}_p_{o}" for o in ORDER]
    t = tbl.dropna(subset=cols + ["rest_h", "rest_a"]).sort_values("date").reset_index(drop=True)
    if len(t) <= min_train:
        return t.iloc[0:0], None
    y, bp, F = t["ftr"].map(Y).to_numpy(), t[cols].to_numpy(float), feature_tensor(t)
    out = np.full((len(t), 3), np.nan)
    theta, cur = None, t["date"].iloc[min_train]
    while cur <= t["date"].max():
        train = (t["date"] < cur).to_numpy()
        test = ((t["date"] >= cur) & (t["date"] < cur + pd.Timedelta(days=step_days))).to_numpy()
        if test.any():
            theta = fit(F[train], bp[train], y[train], lam)
            out[test] = predict(theta, F[test], bp[test])
        cur += pd.Timedelta(days=step_days)
    t["base_p"] = list(bp)
    t[[f"adj_{o}" for o in ORDER]] = out
    return t.dropna(subset=["adj_H"]).reset_index(drop=True), theta


def _ll(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def compare(rows):
    """Per-match log loss difference (base minus adjusted; positive = schedule info helped) with a 95% interval."""
    y = rows["ftr"].map(Y).to_numpy()
    base = np.vstack(rows["base_p"].to_numpy())
    adj = rows[[f"adj_{o}" for o in ORDER]].to_numpy(float)
    d = _ll(base, y) - _ll(adj, y)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"matches": len(d), "base": float(_ll(base, y).mean()), "adjusted": float(_ll(adj, y).mean()),
            "gain": float(d.mean()), "ci95": (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se))}


def bins(tbl):
    """Plain view: after accounting for team strength, does the better-rested side do better than the model expects?"""
    t = tbl.dropna(subset=["rest_h", "rest_a"]).copy()
    lo, hi = REST_CAP
    t["diff"] = np.clip(t["rest_h"], lo, hi) - np.clip(t["rest_a"], lo, hi)
    t["bin"] = pd.cut(t["diff"], [-99, -3, -1, 0, 2, 99], labels=["away rested 3+ days more", "away 1-2 more", "equal", "home 1-2 more", "home 3+ more"])
    y = t["ftr"]
    t["home_gap"] = (y == "H").astype(float) - t["model_H"]
    t["away_gap"] = (y == "A").astype(float) - t["model_A"]
    out = t.groupby("bin", observed=True).agg(matches=("diff", "size"), home_win_vs_model=("home_gap", "mean"), away_win_vs_model=("away_gap", "mean"))
    return out.round(3)


def report(tbl, min_train=1500):
    lines = [f"{len(tbl)} out-of-sample matches in the forecast table."]
    lines.append("\nWHO DOES BETTER THAN THE MODEL EXPECTS? (actual win rate minus model probability; positive = outperformed)")
    lines.append(bins(tbl).to_string())
    for base, label in (("model", "OUR MODEL"), ("avg", "THE MARKET (average bookmaker price)")):
        rows, theta = oos(tbl, base, min_train)
        if rows.empty:
            lines.append(f"\n{label}: not enough matches to fit.")
            continue
        c = compare(rows)
        lines.append(f"\n{label} + rest/congestion, out of sample over {c['matches']} matches:")
        lines.append(f"  log loss {c['base']:.4f} -> {c['adjusted']:.4f}  (gain {c['gain']:+.5f}, 95% interval {c['ci95'][0]:+.5f} to {c['ci95'][1]:+.5f})")
        lines.append("  last fitted weights: " + ", ".join(f"{n} {v:+.3f}" for n, v in zip(NAMES, theta)))
        lines.append("  verdict: " + ("helps" if c["ci95"][0] > 0 else "hurts" if c["ci95"][1] < 0 else "no detectable effect"))
    return "\n".join(lines)
