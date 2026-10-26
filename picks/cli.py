"""python -m picks.cli fetch|backtest|picks"""
import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from . import backtest, data, evaluate
from .model import PoissonModel
from .value import find_picks


XG_WEIGHT = 0.0  # raise only once `evaluate` shows a real-data improvement


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download historical results+odds and upcoming fixtures")
    f.add_argument("--leagues", nargs="+", default=["E0", "E1", "D1", "SP1", "I1", "F1"])
    f.add_argument("--seasons", nargs="+", default=data.season_codes(4))
    sub.add_parser("evaluate", help="compare xG weights and the market on log loss / RPS")
    for name in ("backtest", "picks", "evaluate"):
        s = sub.choices[name] if name == "evaluate" else sub.add_parser(name)
        s.add_argument("--data", default="picks/data")
        s.add_argument("--synthetic", action="store_true", help="use generated data")
        if name != "evaluate":
            s.add_argument("--xg-weight", type=float, default=XG_WEIGHT,
                           help="0 = goals only, 1 = shots-based xG proxy only")
    sub.choices["picks"].add_argument("--out", default="picks.json")
    sub.choices["picks"].add_argument("--fixtures", default="picks/fixtures.csv",
                                      help="football-data fixtures.csv (Div,Date,HomeTeam,AwayTeam,B365H,B365D,B365A)")
    a = ap.parse_args()

    if a.cmd == "fetch":
        data.fetch(a.leagues, a.seasons)
        return data.fetch_fixtures()
    df = data.synthetic() if a.synthetic else data.load(a.data)
    if a.cmd == "evaluate":
        print(evaluate.evaluate(df).to_string())
        return
    if a.cmd == "backtest":
        print(json.dumps(backtest.summarise(backtest.run(df, xg_weight=a.xg_weight)), indent=2))
        return
    models = {lg: PoissonModel(xg_weight=a.xg_weight).fit(g) for lg, g in df.groupby("League")}
    today = datetime.now(timezone.utc).date()
    fx = data.load_fixtures(a.fixtures, leagues=list(models), from_date=today)
    out = []
    for _, m in fx.iterrows():
        model = models[m.League]
        odds = {"H": m.get("B365H"), "D": m.get("B365D"), "A": m.get("B365A")}
        if not (model.knows(m.HomeTeam) and model.knows(m.AwayTeam)) or any(pd.isna(v) for v in odds.values()):
            continue
        for p in find_picks(model.predict(m.HomeTeam, m.AwayTeam), odds):
            out.append({"date": str(m.Date.date()), "league": m.League, "match": f"{m.HomeTeam} v {m.AwayTeam}", **p.__dict__})
    out.sort(key=lambda r: (r["date"], -r["ev"]))
    payload = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "backtest": backtest.summarise(backtest.run(df, xg_weight=a.xg_weight)), "picks": out}
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"{len(fx)} upcoming fixtures checked, {len(out)} picks -> {a.out}")


if __name__ == "__main__":
    main()
