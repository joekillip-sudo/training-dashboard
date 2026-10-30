"""Small logistic regression (no intercept: player order is random, so there is nothing to offset) and a
walk-forward wrapper that refits on earlier matches only."""
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logit(X, y, offset=None, lam=1.0, iters=50):
    """Newton fit of P(y=1) = sigmoid(offset + X@theta), ridge penalty lam*|theta|^2. lam=0 -> plain maximum likelihood."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    theta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(off + X @ theta)
        g = X.T @ (p - y) + 2 * lam * theta
        H = (X * (p * (1 - p))[:, None]).T @ X + 2 * lam * np.eye(X.shape[1]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-9:
            break
    return theta


def slope_with_se(p, y):
    """Calibration slope: fit y ~ slope*logit(p). 1.0 is perfectly calibrated; <1 over-confident; >1 under-confident."""
    x = logit(np.asarray(p, float))[:, None]
    th = fit_logit(x, y, lam=0.0)
    q = sigmoid(x[:, 0] * th[0])
    return float(th[0]), float(1 / np.sqrt((q * (1 - q) * x[:, 0] ** 2).sum()))


def walk_forward_fit(dates, X, y, offset=None, usable=None, test_start="2013-01-01", step_days=90, min_train=2000, lam=1.0):
    """Predict P(player 1 wins) for each row from a model fitted only on rows dated strictly earlier.

    Refits every `step_days` from test_start. Rows with non-finite features are never used. Returns (p, last theta).
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    X = np.asarray(X, float)
    off = np.zeros(len(X)) if offset is None else np.asarray(offset, float)
    ok = np.isfinite(X).all(1) & np.isfinite(off)
    if usable is not None:
        ok &= np.asarray(usable, bool)
    out = np.full(len(X), np.nan)
    theta, cur, end = None, pd.Timestamp(test_start), dates.max()
    while cur <= end:
        train = ((dates < cur) & ok).to_numpy()
        test = ((dates >= cur) & (dates < cur + pd.Timedelta(days=step_days)) & ok).to_numpy()
        if test.any() and train.sum() >= min_train:
            theta = fit_logit(X[train], y[train], off[train], lam)
            out[test] = sigmoid(off[test] + X[test] @ theta)
        cur += pd.Timedelta(days=step_days)
    return out, theta
