"""python -m picks.cli fetch|backtest|picks"""
import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from . import backtest, calibration, data, evaluate, ledger, understat
from .model import PoissonModel
from .value import find_picks


XG_WEIGHT = 0.0  # raise only once `evaluate` shows a real-data improvement


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download historical results+odds and upcoming fixtures")
    f.add_argument("--leagues", nargs="+", default=["E0", "E1", "D1", "SP1", "I1", "F1"])
    f.add_argument("--seasons", nargs="+", default=data.season_codes(4))
    x = sub.add_parser("fetch-xg", help="scrape real xG from Understat (optional; picks/xg_data)")
    x.add_argument("--leagues", nargs="+", default=list(understat.LEAGUES))
    x.add_argument("--seasons", nargs="+", default=data.season_codes(4))
    sub.add_parser("evaluate", help="compare xG weights and the market on log loss / RPS")
    sub.add_parser("calibrate", help="reliability of model probabilities and of the bets taken")
    for name in ("backtest", "picks", "evaluate", "calibrate"):
        s = sub.choices[name] if name in ("evaluate", "calibrate") else sub.add_parser(name)
        s.add_argument("--data", default="picks/data")
        s.add_argument("--synthetic", action="store_true", help="use generated data")
        if name != "evaluate":  # calibrate takes --xg-weight too
            s.add_argument("--xg-weight", type=float, default=XG_WEIGHT,
                           help="0 = goals only, 1 = shots-based xG proxy only")
    sub.choices["picks"].add_argument("--out", default="ledger.json", help="paper-trading ledger (read, updated, written)")
    sub.choices["picks"].add_argument("--seed-backtest", action="store_true",
                                      help="add historical walk-forward bets to the ledger once (idempotent)")
    sub.choices["picks"].add_argument("--fixtures", default="picks/fixtures.csv",
                                      help="football-data fixtures.csv (Div,Date,HomeTeam,AwayTeam,B365H,B365D,B365A)")
    a = ap.parse_args()

    if a.cmd == "fetch-xg":
        if not understat.fetch(a.leagues, a.seasons):
            raise SystemExit("no xG downloaded")
        return
    if a.cmd == "fetch":
        data.fetch(a.leagues, a.seasons)
        return data.fetch_fixtures()
    df = data.synthetic() if a.synthetic else data.load(a.data)
    if a.cmd == "evaluate":
        print(evaluate.evaluate(df).to_string())
        return
    if a.cmd == "calibrate":
        fc = calibration.collect(df, xg_weight=a.xg_weight)
        for src, label in (("model", "MODEL"), ("close", "MARKET CLOSE (Pinnacle)")):
            t = calibration.reliability(fc, src)
            print(f"\n{label}: predicted vs observed, ECE={calibration.ece(t):.4f}\n{t.to_string()}")
        for mkt in ("open", "close"):
            print(f"\n{calibration.blend_log_loss(fc, mkt).to_string()}")
        print("\nBETS TAKEN (by odds band)\n" + calibration.pick_calibration(backtest.run(df, xg_weight=a.xg_weight)).to_string())
        return
    if a.cmd == "backtest":
        print(json.dumps(backtest.summarise(backtest.run(df, xg_weight=a.xg_weight)), indent=2))
        return
    models = {lg: PoissonModel(xg_weight=a.xg_weight).fit(g) for lg, g in df.groupby("League")}
    today = datetime.now(timezone.utc).date()
    book = ledger.load(a.out)
    fx = data.load_fixtures(a.fixtures, leagues=list(models), from_date=today)
    added = ledger.add_live(book, fx, models, today)
    seeded = ledger.seed_backtest(book, df, xg_weight=a.xg_weight) if a.seed_backtest else 0
    settled = ledger.settle(book, df)
    book["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ledger.save(book, a.out)
    print(f"{len(fx)} upcoming fixtures checked: {added} new bets, {settled} settled, {seeded} backtest bets seeded -> {a.out}")


if __name__ == "__main__":
    main()
