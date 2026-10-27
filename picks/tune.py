"""Grid-search the model's settings, scoring on earlier matches and re-checking on later ones.

The pick is made on the early part only; the late part then shows whether it holds up, which a
pick scored on everything could never tell you.
"""
import itertools

import numpy as np
import pandas as pd

from .market import Y, forecast_table

GRID = {
    "half_life_days": (200, 365, 550, 900),
    "shrink": (1.0, 3.0, 8.0),
    "rho": (-0.10, -0.05, 0.0),
}


def _loss(t):
    p = t[["model_H", "model_D", "model_A"]].to_numpy(float)
    y = t["ftr"].map(Y).to_numpy()
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def tune(df, base_kw, defaults, grid=GRID, split=0.65, step_days=14):
    """Returns (table sorted by early log loss, summary dict)."""
    rows, tables = [], {}
    names = list(grid)
    for vals in itertools.product(*(grid[k] for k in names)):
        kw = dict(zip(names, vals))
        t = forecast_table(df, step_days=step_days, **{**base_kw, **kw})
        cut = t["date"].quantile(split)
        rows.append({**kw, "early": round(_loss(t[t["date"] < cut]), 4), "late": round(_loss(t[t["date"] >= cut]), 4), "n": len(t)})
        tables[tuple(vals)] = None
    res = pd.DataFrame(rows).sort_values("early").reset_index(drop=True)
    best = res.iloc[0]
    d = res[(res[names] == pd.Series(defaults)).all(axis=1)]
    summary = {
        "best": {k: float(best[k]) for k in names},
        "best_late": best["late"],
        "default_late": float(d["late"].iloc[0]) if len(d) else None,
    }
    return res, summary
