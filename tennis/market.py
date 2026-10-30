"""De-vigging, flat-stake settling, and uncertainty. Two-outcome version of picks/value.py and picks/market.py."""
import numpy as np
import pandas as pd

Z = 1.96  # 95% margin of error multiplier


def devig(o1, o2):
    """Fair probability of player 1 (proportional margin removal)."""
    a, b = 1 / np.asarray(o1, float), 1 / np.asarray(o2, float)
    return a / (a + b)


def overround(o1, o2):
    return 1 / np.asarray(o1, float) + 1 / np.asarray(o2, float) - 1


def log_loss(p1, y):
    p = np.where(np.asarray(y) == 1, p1, 1 - np.asarray(p1))
    return -np.log(np.clip(p, 1e-12, 1.0))


def mean_moe(x):
    """(n, mean, 95% margin of error) ignoring NaNs."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return 0, np.nan, np.nan
    return n, float(x.mean()), (Z * float(x.std(ddof=1)) / np.sqrt(n) if n > 1 else np.nan)


def settle(p1, o1, o2, y, thr, close_p1=None, max_odds=15.0):
    """One unit on the better-EV side of each match when EV >= thr. Returns (bet mask, profit, odds taken, CLV)."""
    p1, o1, o2, y = (np.asarray(a, float) for a in (p1, o1, o2, y))
    ev1, ev2 = p1 * o1 - 1, (1 - p1) * o2 - 1
    ev1 = np.where(np.isfinite(ev1) & (o1 <= max_odds), ev1, -np.inf)
    ev2 = np.where(np.isfinite(ev2) & (o2 <= max_odds), ev2, -np.inf)
    side1 = ev1 >= ev2
    ev, odds = np.where(side1, ev1, ev2), np.where(side1, o1, o2)
    win = np.where(side1, y == 1, y == 0)
    profit = np.where(win, odds - 1, -1.0)
    clv = np.full(len(y), np.nan)
    if close_p1 is not None:
        cp = np.where(side1, close_p1, 1 - np.asarray(close_p1, float))
        clv = odds * cp - 1
    return ev >= thr, profit, odds, clv


def summarise(bet, profit, odds, clv, late):
    """Bets, hit rate, ROI with margin of error, ROI in the first and second half of the period, mean CLV."""
    n = int(bet.sum())
    if n == 0:
        return {"bets": 0}
    r = {"bets": n, "hit": float((profit[bet] > 0).mean()), "odds": float(odds[bet].mean())}
    _, r["roi"], r["moe"] = mean_moe(profit[bet])
    for name, sel in (("h1", bet & ~late), ("h2", bet & late)):
        k, r[name], r[name + "_moe"] = mean_moe(profit[sel])
        r["n_" + name] = k
    if np.isfinite(clv[bet]).any():
        r["clv"] = float(np.nanmean(clv[bet]))
    return r


def pm(x, moe, scale=100, digits=1):
    """'+2.3 ±4.1' (percent) or '-' when there is nothing."""
    if x is None or not np.isfinite(x):
        return "-"
    s = f"{x * scale:+.{digits}f}"
    return s + (f" ±{moe * scale:.{digits}f}" if moe is not None and np.isfinite(moe) else "")


def paired_gap(ll_model, ll_other):
    """Mean log-loss difference (model minus other) over the same matches. Negative = model better. -> (n, gap, moe)"""
    return mean_moe(np.asarray(ll_model, float) - np.asarray(ll_other, float))


def reliability(p1, y, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)):
    """Calibration of each match's favourite side (p>=0.5), so every match counts once. Observed rate has a 95% margin."""
    p1, y = np.asarray(p1, float), np.asarray(y, float)
    pf = np.where(p1 >= 0.5, p1, 1 - p1)
    wf = np.where(p1 >= 0.5, y, 1 - y)
    rows = []
    cut = pd.cut(pf, list(edges), include_lowest=True)
    for band, idx in pd.Series(np.arange(len(pf))).groupby(cut, observed=True):
        i = idx.to_numpy()
        n, obs = len(i), float(wf[i].mean())
        rows.append({"band": str(band), "n": n, "predicted": float(pf[i].mean()), "observed": obs,
                     "gap": float(pf[i].mean() - obs), "moe": Z * np.sqrt(max(obs * (1 - obs), 1e-9) / n)})
    return pd.DataFrame(rows)
