"""Time-weighted Poisson team-strength model (Dixon-Coles style).

Expected goals: home = attack[h] * defence[a] * home_adv, away = attack[a] * defence[h].
Fitted with weighted fixed-point MLE updates, so it needs only numpy.
"""
import math

import numpy as np
from .xg import shot_xg

MAX_GOALS = 10


class PoissonModel:
    def __init__(self, half_life_days=365, shrink=3.0, rho=-0.05, iters=200, xg_weight=0.0, xg_source="auto"):
        self.xi = math.log(2) / half_life_days
        self.shrink = shrink
        self.rho = rho
        self.iters = iters
        self.xg_weight = xg_weight  # 0 = goals only, 1 = xG only
        self.xg_source = xg_source  # "auto": real xG where present, else shots proxy; "shots": proxy only
        self.attack = {}
        self.defence = {}
        self.home_adv = 1.3
        self.league_avg = 1.3

    def fit(self, matches, as_of=None):
        """matches: DataFrame with Date, HomeTeam, AwayTeam, FTHG, FTAG."""
        df = matches.dropna(subset=["FTHG", "FTAG"])
        as_of = as_of if as_of is not None else df["Date"].max()
        age = (as_of - df["Date"]).dt.days.clip(lower=0).to_numpy()
        w = np.exp(-self.xi * age)

        teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        h = df["HomeTeam"].map(idx).to_numpy()
        a = df["AwayTeam"].map(idx).to_numpy()
        hg = df["FTHG"].to_numpy(float)
        ag = df["FTAG"].to_numpy(float)
        if self.xg_weight > 0:
            hx, ax = self._xg(df)
            if hx is not None:  # blend goals with lower-noise xG where we have it
                wt = self.xg_weight
                hg = np.where(np.isnan(hx), hg, wt * hx + (1 - wt) * hg)
                ag = np.where(np.isnan(ax), ag, wt * ax + (1 - wt) * ag)
        n = len(teams)

        att, dfn, g = np.ones(n), np.ones(n), 1.3
        k = self.shrink
        for _ in range(self.iters):
            # attack: goals scored / exposure
            num = np.bincount(h, w * hg, n) + np.bincount(a, w * ag, n)
            den = np.bincount(h, w * dfn[a] * g, n) + np.bincount(a, w * dfn[h], n)
            att = (num + k) / (den + k)
            att /= att.mean()
            # defence: goals conceded / exposure
            num = np.bincount(a, w * hg, n) + np.bincount(h, w * ag, n)
            den = np.bincount(a, w * att[h] * g, n) + np.bincount(h, w * att[a], n)
            dfn = (num + k) / (den + k)
            dfn /= dfn.mean()
            g = (w * hg).sum() / (w * att[h] * dfn[a]).sum()

        self.attack = dict(zip(teams, att))
        self.defence = dict(zip(teams, dfn))
        self.home_adv = g
        self.league_avg = (w * ag).sum() / (w * att[a] * dfn[h]).sum()
        return self

    def _xg(self, df):
        """Per-match xG: real (Understat) where present, else shots proxy, else NaN."""
        proxy = shot_xg(df)
        hx, ax = (proxy[0], proxy[1]) if proxy is not None else (None, None)
        if self.xg_source == "auto" and "HxG" in df and df["HxG"].notna().any():
            rh, ra = df["HxG"].to_numpy(float), df["AxG"].to_numpy(float)
            hx = rh if hx is None else np.where(np.isnan(rh), hx, rh)
            ax = ra if ax is None else np.where(np.isnan(ra), ax, ra)
        return hx, ax

    def knows(self, team):
        return team in self.attack

    def expected_goals(self, home, away):
        lh = self.attack[home] * self.defence[away] * self.home_adv * self.league_avg
        la = self.attack[away] * self.defence[home] * self.league_avg
        return lh, la

    def score_matrix(self, home, away):
        lh, la = self.expected_goals(home, away)
        pmf = lambda lam: np.array(
            [math.exp(-lam) * lam**i / math.factorial(i) for i in range(MAX_GOALS + 1)]
        )
        m = np.outer(pmf(lh), pmf(la))
        r = self.rho  # Dixon-Coles low-score correction
        m[0, 0] *= 1 - lh * la * r
        m[0, 1] *= 1 + lh * r
        m[1, 0] *= 1 + la * r
        m[1, 1] *= 1 - r
        return m / m.sum()

    def predict(self, home, away):
        """Return dict of 1X2 and over/under 2.5 probabilities."""
        m = self.score_matrix(home, away)
        tot = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        return {
            "H": float(np.tril(m, -1).sum()),
            "D": float(np.trace(m)),
            "A": float(np.triu(m, 1).sum()),
            "O25": float(m[tot > 2].sum()),
            "U25": float(m[tot <= 2].sum()),
        }
