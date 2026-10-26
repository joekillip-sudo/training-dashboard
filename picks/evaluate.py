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


def evaluate(df, xg_weights=(0.0, 0.5, 1.0), **kw):
    """Walk-forward comparison of model variants and the market on the same matches."""
    forecasts, outcomes = {}, {}
    for w in xg_weights:
        rows = {}
        for m, model in walk_forward(df, xg_weight=w, **kw):
            if _has_odds(m):
                pr = model.predict(m.HomeTeam, m.AwayTeam)
                rows[m.name] = {o: pr[o] for o in ORDER}
                outcomes[m.name] = m.FTR
        forecasts[f"model xg_weight={w}"] = rows
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
