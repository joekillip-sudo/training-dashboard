"""Time-weighted Poisson team-strength model (Dixon-Coles style).

Expected goals: home = attack[h] * defence[a] * home_adv, away = attack[a] * defence[h].
Fitted with weighted fixed-point MLE updates, so it needs only numpy.
"""
import math

import numpy as np
import pandas as pd
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
        self.last_seen = {}

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

        # home goals ~ mu * gamma * att[h] * dfn[a]; away goals ~ mu * att[a] * dfn[h].
        # att and dfn are kept at mean 1, so mu is the base scoring rate and gamma the home advantage.
        att, dfn, mu, gam = np.ones(n), np.ones(n), 1.3, 1.3
        k = self.shrink
        pull = k * ((w * hg).sum() + (w * ag).sum()) / (2 * w.sum())  # pseudo-goals shrinking ratings toward 1
        for _ in range(self.iters):
            num = np.bincount(h, w * hg, n) + np.bincount(a, w * ag, n)
            den = np.bincount(h, w * dfn[a] * mu * gam, n) + np.bincount(a, w * dfn[h] * mu, n)
            att = (num + pull) / (den + pull)
            att /= att.mean()
            num = np.bincount(a, w * hg, n) + np.bincount(h, w * ag, n)
            den = np.bincount(a, w * att[h] * mu * gam, n) + np.bincount(h, w * att[a] * mu, n)
            dfn = (num + pull) / (den + pull)
            dfn /= dfn.mean()
            mu = (w * (hg + ag)).sum() / (w * (gam * att[h] * dfn[a] + att[a] * dfn[h])).sum()
            gam = (w * hg).sum() / (mu * (w * att[h] * dfn[a]).sum())

        self.attack = dict(zip(teams, att))
        self.defence = dict(zip(teams, dfn))
        self.home_adv = gam
        self.league_avg = mu
        seen = pd.concat([df[["HomeTeam", "Date"]].set_axis(["team", "date"], axis=1),
                          df[["AwayTeam", "Date"]].set_axis(["team", "date"], axis=1)])
        self.last_seen = seen.groupby("team")["date"].max().to_dict()
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


class MultiLeagueModel:
    """One model for every league.

    Teams in different leagues never play each other, so there is no cross-league strength to
    learn; what each league does need is its own scoring rate and home advantage. This holds one
    fitted PoissonModel per league behind a single fit/predict interface. It also behaves like
    a read-only dict of league -> model, which is how the ledger code looks models up.
    """

    def __init__(self, **model_kw):
        self.model_kw = model_kw
        self.models = {}

    def fit(self, matches, as_of=None):
        self.models = {lg: PoissonModel(**self.model_kw).fit(g, as_of=as_of) for lg, g in matches.groupby("League")}
        return self

    def knows(self, league, team):
        return league in self.models and self.models[league].knows(team)

    def predict(self, league, home, away):
        return self.models[league].predict(home, away)

    def get(self, league, default=None):
        return self.models.get(league, default)

    def items(self):
        return self.models.items()

    def keys(self):
        return self.models.keys()

    def __getitem__(self, league):
        return self.models[league]

    def __contains__(self, league):
        return league in self.models

    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)
