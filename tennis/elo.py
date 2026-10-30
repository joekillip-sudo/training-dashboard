"""Elo ratings, overall and by surface, updated match by match.

A match is predicted from ratings that contain only earlier matches. Matches sharing the same date and round
are predicted first and updated afterwards, so even if the file's dates are tournament start dates nothing
from the same round can leak into a prediction.
"""
import numpy as np
import pandas as pd

SURFACES = ("hard", "clay", "grass")
LN10_400 = np.log(10) / 400


class Elo:
    """Overall + surface Elo. K shrinks as a player accumulates matches (new players move fast)."""

    def __init__(self, k0=250.0, k_off=5.0, k_shape=0.4, w_surface=0.5, init=1500.0):
        self.k0, self.k_off, self.k_shape, self.w, self.init = k0, k_off, k_shape, w_surface, init
        self.ro, self.rs = {}, {}   # overall; (player, surface)
        self.no, self.ns = {}, {}   # match counts

    def _k(self, n):
        return self.k0 / (n + self.k_off) ** self.k_shape

    def _surface(self, a, surf):
        return self.rs.get((a, surf), self.ro.get(a, self.init))

    def rating(self, a, surf):
        ro = self.ro.get(a, self.init)
        return ro if surf not in SURFACES else (1 - self.w) * ro + self.w * self._surface(a, surf)

    def diff(self, a, b, surf):
        """Blended rating difference (a minus b), in Elo points."""
        return self.rating(a, surf) - self.rating(b, surf)

    @staticmethod
    def expected(d):
        return 1.0 / (1.0 + 10.0 ** (-d / 400.0))

    def update(self, win, lose, surf):
        e = self.expected(self.ro.get(win, self.init) - self.ro.get(lose, self.init))
        kw, kl = self._k(self.no.get(win, 0)), self._k(self.no.get(lose, 0))
        sw, sl = (self._surface(win, surf), self._surface(lose, surf)) if surf in SURFACES else (None, None)
        self.ro[win] = self.ro.get(win, self.init) + kw * (1 - e)
        self.ro[lose] = self.ro.get(lose, self.init) - kl * (1 - e)
        self.no[win], self.no[lose] = self.no.get(win, 0) + 1, self.no.get(lose, 0) + 1
        if surf in SURFACES:
            es = self.expected(sw - sl)
            self.rs[(win, surf)] = sw + self._k(self.ns.get((win, surf), 0)) * (1 - es)
            self.rs[(lose, surf)] = sl - self._k(self.ns.get((lose, surf), 0)) * (1 - es)
            self.ns[(win, surf)] = self.ns.get((win, surf), 0) + 1
            self.ns[(lose, surf)] = self.ns.get((lose, surf), 0) + 1


def run_elo(m, **params):
    """Walk through matches in time order. Adds elo_diff (player 1 minus 2), elo_p (player 1 wins), n1, n2 (prior matches).

    `m` must be sorted by (date, rround), as `data.orient` returns it. y is read only after the group is predicted.
    """
    m = m.copy()
    n = len(m)
    keys = list(zip(m["date"], m["rround"]))
    p1, p2, surf, tour, y = (m[c].to_numpy(object) for c in ("p1", "p2", "surface", "tour", "y"))
    diff, n1, n2 = np.zeros(n), np.zeros(n, int), np.zeros(n, int)
    elo = Elo(**params)
    i = 0
    while i < n:
        j = i
        while j < n and keys[j] == keys[i]:
            j += 1
        for k in range(i, j):
            a, b = (tour[k], p1[k]), (tour[k], p2[k])
            diff[k] = elo.diff(a, b, surf[k])
            n1[k], n2[k] = elo.no.get(a, 0), elo.no.get(b, 0)
        for k in range(i, j):
            a, b = (tour[k], p1[k]), (tour[k], p2[k])
            w, l = (a, b) if y[k] == 1 else (b, a)
            elo.update(w, l, surf[k])
        i = j
    m["elo_diff"], m["n1"], m["n2"] = diff, n1, n2
    m["elo_p"] = Elo.expected(diff)
    return m, elo
