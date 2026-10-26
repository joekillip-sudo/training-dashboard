"""Shots-based expected-goals proxy.

football-data.co.uk has shots (HS/AS) and shots on target (HST/AST) but no true xG. Goals are
noisy; shots are a steadier signal of how well a team really creates chances. We fit
goals ~ a*on_target + b*off_target on the training data only (no look-ahead) and use the
prediction as a lower-noise stand-in for goals.
"""
import numpy as np

SHOT_COLS = ("HS", "AS", "HST", "AST")
MIN_ROWS = 100


def shot_xg(df):
    """Return (home_xg, away_xg, coefficients), or None if shot data is missing/too thin.

    xg arrays are NaN for matches without shot data.
    """
    if not all(c in df.columns for c in SHOT_COLS):
        return None
    ok = df[list(SHOT_COLS) + ["FTHG", "FTAG"]].dropna()
    if len(ok) < MIN_ROWS:
        return None
    X = np.vstack([
        np.column_stack([ok["HST"], ok["HS"] - ok["HST"]]),
        np.column_stack([ok["AST"], ok["AS"] - ok["AST"]]),
    ])
    y = np.concatenate([ok["FTHG"], ok["FTAG"]])
    coef = np.clip(np.linalg.lstsq(X, y, rcond=None)[0], 0, None)
    home = coef[0] * df["HST"] + coef[1] * (df["HS"] - df["HST"])
    away = coef[0] * df["AST"] + coef[1] * (df["AS"] - df["AST"])
    return home.to_numpy(float), away.to_numpy(float), coef
