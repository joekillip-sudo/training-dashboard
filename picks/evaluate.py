"""Score probability forecasts against what happened, with the bookmakers as the benchmark.

Lower is better for both metrics. A model that can't beat the market's closing prices on
these is very unlikely to profit; profit/ROI alone is far too noisy to tell.
"""
import numpy as np
import pandas as pd

from .backtest import CLOSE, OPEN, walk_forward
from .value import devig

ORDER = ("H", "D", "A")


def log_loss(p, outcome):
    return -np.log(max(p[outcome], 1e-12))


def rps(p, outcome):
    """Ranked probability score over the ordered outcomes H < D < A."""
    cum_p = np.cumsum([p[o] for o in ORDER])
    cum_o = np.cumsum([1.0 if o == outcome else 0.0 for o in ORDER])
    return float(((cum_p - cum_o)[:-1] ** 2).sum() / 2)


def _has_odds(m):
    return all(not pd.isna(m.get(c)) for c in [*OPEN.values(), *CLOSE.values()])


def default_variants(df):
    """(xg_weight, xg_source) pairs; real xG variants only when the data has real xG."""
    v = [(0.0, "shots")] + [(w, "shots") for w in (0.5, 1.0)]
    if "HxG" in df and df["HxG"].notna().any():
        v += [(w, "auto") for w in (0.5, 1.0)]
    return v


def evaluate(df, variants=None, **kw):
    """Walk-forward comparison of model variants and the market on the same matches."""
    forecasts, outcomes = {}, {}
    for w, src in variants or default_variants(df):
        rows = {}
        for m, model in walk_forward(df, xg_weight=w, xg_source=src, **kw):
            if _has_odds(m):
                pr = model.predict(m.HomeTeam, m.AwayTeam)
                rows[m.name] = {o: pr[o] for o in ORDER}
                outcomes[m.name] = m.FTR
        label = "goals only" if w == 0 else f"xg={w} {'real+shots' if src == 'auto' else 'shots proxy'}"
        forecasts[f"model {label}"] = rows
    common = set.intersection(*(set(r) for r in forecasts.values())) if forecasts else set()
    forecasts["market open (Bet365)"] = {i: devig({o: df.at[i, c] for o, c in OPEN.items()}) for i in common}
    forecasts["market close (Pinnacle)"] = {i: devig({o: df.at[i, c] for o, c in CLOSE.items()}) for i in common}

    table = {}
    for name, rows in forecasts.items():
        ids = sorted(common)
        table[name] = {
            "matches": len(ids),
            "log_loss": round(float(np.mean([log_loss(rows[i], outcomes[i]) for i in ids])), 4),
            "rps": round(float(np.mean([rps(rows[i], outcomes[i]) for i in ids])), 4),
        } if ids else {"matches": 0}
    return pd.DataFrame(table).T
