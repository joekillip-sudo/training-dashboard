"""Walk-forward backtest: refit before each match week, bet only on past information."""
import numpy as np
import pandas as pd

from .model import PoissonModel
from .value import devig, find_picks

OPEN = {"H": "B365H", "D": "B365D", "A": "B365A"}
CLOSE = {"H": "PSCH", "D": "PSCD", "A": "PSCA"}


def run(df, warmup_days=300, step_days=7, min_ev=0.05, min_edge=0.03, kelly=0.25, **model_kw):
    start = df["Date"].min() + pd.Timedelta(days=warmup_days)
    bets, bank = [], 1.0
    t = start
    while t <= df["Date"].max():
        train = df[df["Date"] < t]
        window = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=step_days))]
        t += pd.Timedelta(days=step_days)
        if window.empty:
            continue
        model = PoissonModel(**model_kw).fit(train, as_of=t - pd.Timedelta(days=step_days))
        for _, m in window.iterrows():
            if not (model.knows(m.HomeTeam) and model.knows(m.AwayTeam)):
                continue
            if any(pd.isna(m.get(c)) for c in OPEN.values()):
                continue
            odds = {o: m[c] for o, c in OPEN.items()}
            for p in find_picks(model.predict(m.HomeTeam, m.AwayTeam), odds, min_ev, min_edge, kelly=kelly):
                stake = p.stake * bank
                win = m.FTR == p.outcome
                profit = stake * (p.odds - 1) if win else -stake
                bank += profit
                clv = np.nan
                if all(c in m and not pd.isna(m[c]) for c in CLOSE.values()):
                    close_p = devig({o: m[c] for o, c in CLOSE.items()})[p.outcome]
                    clv = p.odds * close_p - 1  # >0: we beat the closing price
                bets.append({"date": m.Date, "match": f"{m.HomeTeam} v {m.AwayTeam}", "pick": p.outcome,
                             "odds": p.odds, "model_p": p.model_p, "ev": p.ev, "stake": stake,
                             "won": win, "profit": profit, "clv": clv, "bank": bank})
    return pd.DataFrame(bets)


def summarise(bets):
    if bets.empty:
        return {"bets": 0}
    staked = bets["stake"].sum()
    return {
        "bets": len(bets),
        "hit_rate": round(bets["won"].mean(), 3),
        "roi_on_staked": round(bets["profit"].sum() / staked, 4),
        "final_bankroll": round(bets["bank"].iloc[-1], 3),
        "avg_clv": round(bets["clv"].mean(), 4),
        "max_drawdown": round((bets["bank"] / bets["bank"].cummax() - 1).min(), 3),
    }
