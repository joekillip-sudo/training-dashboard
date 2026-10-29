"""python -m picks.cli fetch|backtest|picks"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import backtest, calibration, data, evaluate, ledger, market, schedule, tune, understat
from .model import MultiLeagueModel, PoissonModel
from .value import find_picks


XG_WEIGHT = 0.5  # real-data `evaluate` (3,445 matches): log loss 1.0016 goals-only -> 0.9958 at 0.5


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download historical results+odds and upcoming fixtures")
    f.add_argument("--leagues", nargs="+", default=data.ALL_LEAGUES)
    f.add_argument("--seasons", nargs="+", default=data.season_codes(4))
    f.add_argument("--out", default="picks/data", help="where to save the CSVs")
    f.add_argument("--no-fixtures", action="store_true", help="skip the upcoming-fixtures download")
    x = sub.add_parser("fetch-xg", help="scrape real xG from Understat (optional; picks/xg_data)")
    x.add_argument("--leagues", nargs="+", default=list(understat.LEAGUES))
    x.add_argument("--seasons", nargs="+", default=data.season_codes(4))
    sub.add_parser("evaluate", help="compare xG weights and the market on log loss / RPS")
    sub.add_parser("strategies", help="market-anchored model and best-price betting vs the old approach")
    sub.add_parser("schedule", help="does rest / fixture congestion add anything to the model or the market?")
    sub.add_parser("leagues", help="where is the market softer? model vs market, league by league")
    sub.add_parser("tune", help="grid-search decay/shrinkage/rho with a held-out check")
    sub.add_parser("calibrate", help="reliability of model probabilities and of the bets taken")
    for name in ("backtest", "picks", "evaluate", "calibrate", "strategies", "tune", "leagues", "schedule"):
        s = sub.choices[name] if name in ("evaluate", "calibrate", "strategies", "tune", "leagues", "schedule") else sub.add_parser(name)
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
        data.fetch(a.leagues, a.seasons, a.out)
        return None if a.no_fixtures else data.fetch_fixtures()
    df = data.synthetic() if a.synthetic else data.load(a.data)
    if a.cmd == "evaluate":
        print(evaluate.evaluate(df).to_string())
        return
    if a.cmd == "schedule":
        df = schedule.add_schedule_features(df)
        tbl = market.forecast_table(df, xg_weight=a.xg_weight)
        print(schedule.report(tbl, min_train=min(1500, len(tbl) // 3)))
        return
    if a.cmd == "leagues":
        tbl = market.forecast_table(df, xg_weight=a.xg_weight)
        fc, bt = market.league_report(tbl)
        print(f"{len(tbl)} out-of-sample matches across {tbl['league'].nunique()} leagues (model fitted on earlier matches only)")
        print("\nFORECAST QUALITY vs the average bookmaker price (log loss; gap < 0 means the model beats the market):")
        print(fc.to_string(index=False))
        print("\nBETTING (one unit on the best-EV outcome when EV >= 5%; clv vs closing price; early/late = halves of the period):")
        print("  old = model vs Bet365 | best = model vs best price across bookmakers")
        print(bt.to_string(index=False))
        return
    if a.cmd == "strategies":
        tbl = market.forecast_table(df, xg_weight=a.xg_weight)
        oos, theta = market.residual_oos(tbl, min_train=min(1000, len(tbl) // 3))
        print(f"out-of-sample matches: {len(oos)} (of {len(tbl)}); residual theta [draw, away, model-vs-base, base tilt] = "
              f"{None if theta is None else np.round(theta, 3).tolist()}")
        if oos.empty:
            print("not enough matches with Pinnacle pre-match prices to fit the residual model")
            return
        print("\nlog loss on these matches (lower is better):", market.log_losses(oos))
        print("\nFlat one-unit bets on the best-EV outcome per match; early/late = first/second half of the period:")
        print(market.compare(oos).to_string(index=False))
        return
    if a.cmd == "tune":
        res, summary = tune.tune(df, {"xg_weight": a.xg_weight}, {"half_life_days": 365, "shrink": 3.0, "rho": -0.05})
        print("top settings by early log loss (model forecasts vs actual results):")
        print(res.head(8).to_string(index=False))
        print(f"\nbest {summary['best']}: late {summary['best_late']:.4f} vs defaults late {summary['default_late']}")
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
    models = MultiLeagueModel(xg_weight=a.xg_weight).fit(df)
    now = datetime.now(timezone.utc)
    today = now.date()
    book = ledger.load(a.out)
    raw = data.load_fixtures(a.fixtures)
    print(f"fixtures file: {len(raw)} rows, {raw['Date'].min().date()} to {raw['Date'].max().date()}; leagues {sorted(raw['League'].unique()) if 'League' in raw else 'n/a'}")
    fx = data.load_fixtures(a.fixtures, leagues=list(models), from_date=today)
    if "League" not in fx.columns:  # fixtures file has no division column
        fx = data.infer_leagues(fx, models)
    added = ledger.add_live(book, fx, models, now)
    seeded = ledger.seed_backtest(book, df, xg_weight=a.xg_weight) if a.seed_backtest else 0
    settled = ledger.settle(book, df)
    book["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ledger.save(book, a.out)
    print(f"{len(fx)} upcoming fixtures checked: {added} new bets, {settled} settled, {seeded} backtest bets seeded -> {a.out}")


if __name__ == "__main__":
    main()
