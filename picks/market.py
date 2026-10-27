"""Market-anchored forecasts and betting strategies.

Pinnacle's pre-match line is the sharpest price we can see before kickoff, so it is the base.
A small regularised correction learns what the base misses (the model's disagreement with it,
and the market's own favourite/draw biases). Every correction is fitted on earlier matches only.
"""
import numpy as np
import pandas as pd

from .backtest import walk_forward
from .value import devig

ORDER = ("H", "D", "A")
SOURCES = {
    "ps": ("PSH", "PSD", "PSA"),          # Pinnacle, pre-match
    "avg": ("AvgH", "AvgD", "AvgA"),      # average across bookmakers, pre-match
    "b365": ("B365H", "B365D", "B365A"),
    "max": ("MaxH", "MaxD", "MaxA"),      # best price on offer, pre-match
    "close": ("PSCH", "PSCD", "PSCA"),    # Pinnacle closing line: benchmark only, not known at bet time
}
Y = {"H": 0, "D": 1, "A": 2}


def forecast_table(df, **model_kw):
    """Walk-forward model probabilities beside every price source (probabilities de-vigged)."""
    rows = []
    for m, model in walk_forward(df, **model_kw):
        pr = model.predict(m.HomeTeam, m.AwayTeam)
        r = {"date": m.Date, "league": m.League, "ftr": m.FTR}
        for o in ORDER:
            r[f"model_{o}"] = pr[o]
        for name, cols in SOURCES.items():
            vals = [m.get(c) for c in cols]
            ok = all(pd.notna(v) and v > 1 for v in vals)
            fair = devig(dict(zip(ORDER, vals))) if ok else {}
            for o, v in zip(ORDER, vals):
                r[f"{name}_odds_{o}"] = float(v) if ok else np.nan
                r[f"{name}_p_{o}"] = fair.get(o, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def _cols(t, name, kind="p"):
    return t[[f"{name}_{kind}_{o}" if kind == "p" else f"{name}_odds_{o}" for o in ORDER]].to_numpy(float)


def _features(model_p, base_p):
    lb = np.log(base_p)
    lm = np.log(np.clip(model_p, 1e-6, 1.0))
    n = len(base_p)
    F = np.zeros((n, 3, 4))
    F[:, 1, 0] = 1.0           # draw intercept
    F[:, 2, 1] = 1.0           # away intercept
    F[:, :, 2] = lm - lb       # where the model disagrees with the base
    F[:, :, 3] = lb            # favourite/longshot tilt of the base itself
    return F, lb


def fit_residual(model_p, base_p, y, lam=20.0, iters=30):
    """Newton fit of p ∝ base * exp(theta·f), shrunk toward theta=0 (which returns the base unchanged)."""
    F, off = _features(model_p, base_p)
    Yoh = np.eye(3)[y]
    theta = np.zeros(4)
    for _ in range(iters):
        s = off + F @ theta
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(1, keepdims=True)
        Ef = (p[:, :, None] * F).sum(1)
        g = (Ef - (Yoh[:, :, None] * F).sum(1)).sum(0) + 2 * lam * theta
        Eff = (p[:, :, None, None] * F[:, :, :, None] * F[:, :, None, :]).sum(1)
        H = (Eff - Ef[:, :, None] * Ef[:, None, :]).sum(0) + 2 * lam * np.eye(4)
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-7:
            break
    return theta


def predict_residual(theta, model_p, base_p):
    F, off = _features(model_p, base_p)
    s = off + F @ theta
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(1, keepdims=True)


def residual_oos(tbl, base="ps", min_train=1000, step_days=30, lam=20.0):
    """Refit monthly on everything earlier, predict the next month. Returns (rows, last theta).

    Rows before the first fit, or without a base price, are dropped, so all comparisons use the
    same out-of-sample matches.
    """
    need = [f"{base}_p_{o}" for o in ORDER]
    t = tbl.dropna(subset=need).sort_values("date").reset_index(drop=True)
    if len(t) <= min_train:
        return t.iloc[0:0], None
    y = t["ftr"].map(Y).to_numpy()
    mp, bp = t[[f"model_{o}" for o in ORDER]].to_numpy(float), t[need].to_numpy(float)
    out = np.full((len(t), 3), np.nan)
    theta = None
    cur = t["date"].iloc[min_train]
    while cur <= t["date"].max():
        train = (t["date"] < cur).to_numpy()
        test = ((t["date"] >= cur) & (t["date"] < cur + pd.Timedelta(days=step_days))).to_numpy()
        if test.any():
            theta = fit_residual(mp[train], bp[train], y[train], lam)
            out[test] = predict_residual(theta, mp[test], bp[test])
        cur += pd.Timedelta(days=step_days)
    t[[f"resid_p_{o}" for o in ORDER]] = out
    return t.dropna(subset=["resid_p_H"]).reset_index(drop=True), theta


def settle(p, odds, y, close_p=None, thr=0.0, max_odds=10.0):
    """One unit on the highest-EV outcome per match when EV >= thr. Returns summary dict plus per-bet profit."""
    ev = p * odds - 1
    ev = np.where(np.isfinite(ev) & (odds <= max_odds), ev, -np.inf)
    best = ev.argmax(1)
    idx = np.arange(len(y))
    bet = ev[idx, best] >= thr
    win = best == y
    profit = np.where(win, odds[idx, best] - 1, -1.0)
    clv = np.nan * np.ones(len(y))
    if close_p is not None:
        clv = odds[idx, best] * close_p[idx, best] - 1
    return bet, profit, clv


def compare(t, thresholds=(0.0, 0.01, 0.02, 0.03, 0.05)):
    """Strategies side by side on the same out-of-sample matches (one-unit flat stakes)."""
    y = t["ftr"].map(Y).to_numpy()
    close = _cols(t, "close") if "close_p_H" in t else None
    model, ps, avg = (t[[f"model_{o}" for o in ORDER]].to_numpy(float), _cols(t, "ps"), _cols(t, "avg"))
    resid = t[[f"resid_p_{o}" for o in ORDER]].to_numpy(float)
    b365, mx = _cols(t, "b365", "odds"), _cols(t, "max", "odds")
    strategies = {
        "model vs Bet365 (the old approach)": (model, b365),
        "Pinnacle fair price vs Bet365": (ps, b365),
        "Pinnacle fair price vs best price": (ps, mx),
        "average fair price vs best price": (avg, mx),
        "residual model vs Bet365": (resid, b365),
        "residual model vs best price": (resid, mx),
    }
    half = t["date"] >= t["date"].median()
    rows = []
    for name, (p, odds) in strategies.items():
        for thr in thresholds:
            bet, profit, clv = settle(p, odds, y, close, thr)
            n = int(bet.sum())
            if not n:
                rows.append({"strategy": name, "min_ev": thr, "bets": 0})
                continue
            h1, h2 = bet & ~half.to_numpy(), bet & half.to_numpy()
            rows.append({
                "strategy": name, "min_ev": thr, "bets": n,
                "hit": round(float((profit[bet] > 0).mean()), 3),
                "roi": round(float(profit[bet].mean()), 4),
                "clv": round(float(np.nanmean(clv[bet])), 4) if close is not None and np.isfinite(clv[bet]).any() else None,
                "roi_early": round(float(profit[h1].mean()), 4) if h1.any() else None,
                "roi_late": round(float(profit[h2].mean()), 4) if h2.any() else None,
            })
    return pd.DataFrame(rows)


def log_losses(t):
    y = t["ftr"].map(Y).to_numpy()
    idx = np.arange(len(y))
    out = {}
    for name in ("model", "resid", "ps", "avg", "b365", "close"):
        cols = [f"model_{o}" if name == "model" else f"{name}_p_{o}" for o in ORDER]
        if cols[0] not in t:
            continue
        p = t[cols].to_numpy(float)
        out[name] = round(float(-np.log(np.clip(p[idx, y], 1e-12, 1)).mean()), 4)
    return out
