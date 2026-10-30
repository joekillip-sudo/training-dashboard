"""Walk-forward forecasts for every match, beside every price source (probabilities de-vigged)."""
import numpy as np
import pandas as pd

from .elo import LN10_400, run_elo
from .fit import logit, walk_forward_fit
from .market import devig

MIN_PRIOR = 10  # a player with fewer earlier matches has no meaningful rating; those matches are left out of the tests
SOURCES = ("ps", "avg", "max", "b365", "psc")


def _logratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(b / a)
    return np.where(np.isfinite(r), r, 0.0)  # missing/zero rank or points -> no information


def forecast_frame(m, test_start="2013-01-01", elo_params=None, lam_b=1.0, lam_c=20.0):
    """m: oriented matches (data.orient). Adds:
        pA  Elo only (overall+surface blend)
        pB  logistic stack of Elo, ranking and ranking points, refitted every 90 days on EARLIER matches
        pC  Pinnacle's fair price nudged by what Elo says that Pinnacle misses (shrunk to the market), same refit rule
        <src>_p1  de-vigged fair probability of player 1 for each price source
    Rows before test_start (the warm-up) get NaN for pB and pC.
    """
    f, _ = run_elo(m, **(elo_params or {}))
    f["pA"] = f["elo_p"]
    for s in SOURCES:
        f[f"{s}_p1"] = devig(f[f"{s}_o1"], f[f"{s}_o2"])
    warm = ((f["n1"] >= MIN_PRIOR) & (f["n2"] >= MIN_PRIOR)).to_numpy()
    f["warm"] = warm
    elo_logit = f["elo_diff"].to_numpy() * LN10_400
    X = np.column_stack([elo_logit, _logratio(f["r1"], f["r2"]), _logratio(f["pts2"], f["pts1"])])
    y = f["y"].to_numpy()
    f["pB"], f.attrs["theta_B"] = walk_forward_fit(f["date"], X, y, usable=warm, test_start=test_start, lam=lam_b)
    ps = f["ps_p1"].to_numpy()
    off = logit(ps)
    Xc = np.column_stack([elo_logit - off, off])
    f["pC"], f.attrs["theta_C"] = walk_forward_fit(f["date"], Xc, y, offset=off, usable=warm, test_start=test_start, lam=lam_c, min_train=1500)
    return f


def test_sample(f, test_start="2013-01-01", need=("ps", "avg", "max")):
    """The out-of-sample matches that every comparison uses: warm players, all listed prices present, model forecasts present."""
    ok = (f["date"] >= pd.Timestamp(test_start)) & f["warm"] & f["pB"].notna()
    for s in need:
        ok &= f[f"{s}_o1"].notna()
    if f["pC"].notna().any():
        ok &= f["pC"].notna()
    return f[ok].reset_index(drop=True)
