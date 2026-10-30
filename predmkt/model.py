"""Forecast -> probability distribution of the day's (whole-degree) high -> probability of each bracket.

mu = forecast + bias; the high is a whole number, modelled as Normal(mu, sigma) rounded to the nearest degree.
bias and sigma come only from the trailing residuals (actual - forecast) of days strictly before the cut-off.
"""
import math

import numpy as np

SIGMA_FLOOR = 1.0
MIN_TRAIN = 30


def ncdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cdf_le(k, mu, sigma):
    """P(high <= k) for a whole-number high (k may be +-inf)."""
    if k == math.inf:
        return 1.0
    if k == -math.inf:
        return 0.0
    return ncdf((k + 0.5 - mu) / sigma)


def bracket_prob(lo, hi, mu, sigma):
    """P(lo <= high <= hi), lo/hi inclusive whole degrees or +-inf."""
    lower = 0.0 if lo == -math.inf else cdf_le(lo - 1, mu, sigma)
    return cdf_le(hi, mu, sigma) - lower


def fit_bias_sigma(forecast, actual):
    """Bias and spread of (actual - forecast). Needs MIN_TRAIN pairs; sample sd with the floor applied once."""
    r = np.asarray(actual, float) - np.asarray(forecast, float)
    if len(r) < MIN_TRAIN:
        return None
    return float(r.mean()), float(max(r.std(ddof=1), SIGMA_FLOOR))


def pit(y, mu, sigma, u=0.5):
    """Randomised probability-integral transform of a whole-number outcome (u in [0,1) breaks the tie)."""
    lo, hi = cdf_le(y - 1, mu, sigma), cdf_le(y, mu, sigma)
    return lo + u * (hi - lo)
