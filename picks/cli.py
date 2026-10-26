"""python -m picks.cli fetch|backtest|picks"""
import argparse
import json

import pandas as pd

from . import backtest, data
from .model import PoissonModel
from .value import find_picks


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--leagues", nargs="+", default=["E0", "E1", "D1", "SP1", "I1", "F1"])
    f.add_argument("--seasons", nargs="+", default=["2223", "2324", "2425", "2526"])
    for name in ("backtest", "picks"):
        s = sub.add_parser(name)
        s.add_argument("--data", default="picks/data")
        s.add_argument("--synthetic", action="store_true", help="use generated data")
    sub.choices["picks"].add_argument("--out", default="picks.json")
    sub.choices["picks"].add_argument("--fixtures", help="CSV: Date,HomeTeam,AwayTeam,B365H,B365D,B365A")
    a = ap.parse_args()

    if a.cmd == "fetch":
        return data.fetch(a.leagues, a.seasons)
    df = data.synthetic() if a.synthetic else data.load(a.data)
    if a.cmd == "backtest":
        bets = backtest.run(df)
        print(json.dumps(backtest.summarise(bets), indent=2))
        return
    model = PoissonModel().fit(df)
    fx = pd.read_csv(a.fixtures, parse_dates=["Date"]) if a.fixtures else df.tail(0)
    out = []
    for _, m in fx.iterrows():
        if not (model.knows(m.HomeTeam) and model.knows(m.AwayTeam)):
            continue
        odds = {"H": m.B365H, "D": m.B365D, "A": m.B365A}
        for p in find_picks(model.predict(m.HomeTeam, m.AwayTeam), odds):
            out.append({"date": str(m.Date.date()), "match": f"{m.HomeTeam} v {m.AwayTeam}", **p.__dict__})
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{len(out)} picks -> {a.out}")


if __name__ == "__main__":
    main()
