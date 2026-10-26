# Football value-picks engine

Finds matches where a statistical model rates a team higher than the bookmaker's price implies.

## How it works
1. `model.py` fits a time-decayed Poisson model (attack/defence per team, home advantage, Dixon-Coles low-score fix) and outputs 1X2 and over/under 2.5 probabilities.
2. `value.py` strips the bookmaker margin (`devig`), compares fair market probability to model probability, and keeps bets with edge and EV above thresholds. Stakes use capped fractional Kelly.
3. `backtest.py` walks forward week by week (fit on the past only), bets at Bet365 odds, and reports ROI, drawdown and **closing line value** against Pinnacle closing odds.

## Usage
    python -m picks.cli fetch                 # download football-data.co.uk CSVs to picks/data
    python -m picks.cli backtest              # walk-forward backtest on real data
    python -m picks.cli backtest --synthetic  # pipeline check without data
    python -m picks.cli picks --fixtures upcoming.csv --out picks.json
    python -m unittest picks.tests.test_picks

## Reading the results
Judge the model by average CLV first: if it does not consistently beat the closing price, ROI over a few hundred bets is mostly luck. Synthetic results only prove the plumbing.
