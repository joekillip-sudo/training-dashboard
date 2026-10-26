# Football value-picks engine

Finds matches where a statistical model rates a team higher than the bookmaker's price implies.

## How it works
1. `model.py` fits a time-decayed Poisson model (attack/defence per team, home advantage, Dixon-Coles low-score fix) and outputs 1X2 and over/under 2.5 probabilities.
2. `value.py` strips the bookmaker margin (`devig`), compares fair market probability to model probability, and keeps bets with edge and EV above thresholds. Stakes use capped fractional Kelly.
3. `backtest.py` walks forward week by week (fit on the past only), bets at Bet365 odds, and reports ROI, drawdown and **closing line value** against Pinnacle closing odds.

## Usage
    pip install -r picks/requirements.txt
    python -m picks.cli fetch                 # last 4 seasons of results+odds, plus upcoming fixtures
    python -m picks.cli backtest              # walk-forward backtest on real data
    python -m picks.cli picks                 # writes picks.json from picks/fixtures.csv
    python -m picks.cli backtest --synthetic  # pipeline check without data
    python -m unittest picks.tests.test_picks

Models are fitted per league, since leagues score at different rates.

## xG
football-data has no true xG, so `xg.py` builds a proxy from shots and shots on target (goals ~ a*on_target + b*off_target, fitted on training data only). `--xg-weight` (0 to 1) blends it with real goals when fitting team strengths.

    python -m picks.cli evaluate              # log loss / RPS for xg_weight 0, 0.5, 1 vs Bet365 and Pinnacle
    python -m picks.cli backtest --xg-weight 0.5

Default is 0 (goals only). Set `XG_WEIGHT` in `cli.py` only after `evaluate` on real data shows a lower log loss than 0, and ideally closer to Pinnacle's closing prices. Real xG (Understat/FBref) would be a further step but needs scraping.

## Calibration
    python -m picks.cli calibrate             # add --xg-weight to test a variant

Prints, walk-forward: (1) predicted vs observed frequency by probability band for the model and for Pinnacle's close, with the expected calibration error (ECE); (2) log loss of `(1-a)*model + a*market` for a = 0..1, where the lowest point is the best model/market mix; (3) the bets actually taken, by odds band, showing model win chance vs hit rate.

How to read it: a positive `gap` means the model is overconfident in that band. If the best blend sits well above a=0, the model is adding noise relative to the market and bets should lean on the market price. If bets in the 3.0+ odds bands show large overconfidence and negative ROI, the "value" there is mostly model error.

## Automation
`.github/workflows/update_picks.yml` runs daily at 08:17 UTC (and on demand via *Run workflow*): tests, fetch, picks, then commits `picks.json` if it changed. GitHub only runs scheduled workflows from the default branch, so merge to `main` for the schedule to start.

Odds are a snapshot at run time and move before kick-off, so re-check the price before betting.

## Reading the results
Judge the model by average CLV first: if it does not consistently beat the closing price, ROI over a few hundred bets is mostly luck. Synthetic results only prove the plumbing.
