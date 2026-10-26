"""Is the model's 20% really 20%? Reliability checks on forecasts and on the bets taken.

Overconfidence on longshots is the classic way a goals model "finds" value that isn't there:
the model rates an underdog higher than the market, but those extra wins never arrive.
"""
import numpy as np
import pandas as pd

from .backtest import CLOSE, OPEN, walk_forward
from .evaluate import ORDER, _has_odds
from .value import devig

BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]


def collect(df, **kw):
    """Walk-forward model probabilities beside open/close market probabilities, per match."""
    rows = []
    for m, model in walk_forward(df, **kw):
        if not _has_odds(m):
            continue
        pr = model.predict(m.HomeTeam, m.AwayTeam)
        op = devig({o: m[c] for o, c in OPEN.items()})
        cl = devig({o: m[c] for o, c in CLOSE.items()})
        row = {"league": m.League, "ftr": m.FTR}
        for o in ORDER:
            row[f"model_{o}"], row[f"open_{o}"], row[f"close_{o}"] = pr[o], op[o], cl[o]
        rows.append(row)
    return pd.DataFrame(rows)


def reliability(fc, source="model", bins=BINS):
    """Pool H/D/A forecasts into probability bins: predicted vs observed frequency."""
    p = np.concatenate([fc[f"{source}_{o}"].to_numpy() for o in ORDER])
    y = np.concatenate([(fc["ftr"] == o).to_numpy(float) for o in ORDER])
    b = pd.cut(p, bins, include_lowest=True)
    t = pd.DataFrame({"p": p, "y": y, "bin": b}).groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    t["gap"] = t["predicted"] - t["observed"]  # >0 means overconfident in this band
    return t.round(3)


def ece(table):
    """Expected calibration error: bin gaps weighted by how many forecasts fall in each."""
    return float((table["gap"].abs() * table["n"]).sum() / table["n"].sum())


def blend_log_loss(fc, market="close", weights=np.linspace(0, 1, 11)):
    """Log loss of (1-a)*model + a*market. a=0 is the pure model, a=1 the pure market."""
    out = {}
    for a in weights:
        ll = []
        for o in ORDER:
            sel = fc["ftr"] == o
            p = (1 - a) * fc.loc[sel, f"model_{o}"] + a * fc.loc[sel, f"{market}_{o}"]
            ll.append(-np.log(p.clip(lower=1e-12)))
        out[round(float(a), 1)] = round(float(pd.concat(ll).mean()), 4)
    return pd.Series(out, name=f"log_loss_vs_{market}")


def pick_calibration(bets, edges=(1.5, 2.0, 3.0, 5.0, 100)):
    """Among bets actually placed: did the model's stated win chance materialise? By odds band."""
    if bets.empty:
        return pd.DataFrame()
    band = pd.cut(bets["odds"], [1.0, *edges], include_lowest=True)
    g = bets.groupby(band, observed=True)
    t = g.agg(bets=("won", "size"), model_p=("model_p", "mean"), hit_rate=("won", "mean"),
              staked=("stake", "sum"), profit=("profit", "sum"))
    t["roi"] = t["profit"] / t["staked"]
    t["overconfidence"] = t["model_p"] - t["hit_rate"]
    return t.drop(columns=["staked", "profit"]).round(3)
