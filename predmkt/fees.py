"""Kalshi trading fee. Taker fee = ceil_to_cent(0.07 * contracts * P * (1 - P)); settlement is free.

The series metadata confirms fee_type 'quadratic', multiplier 1 (probe report). The 0.07 coefficient comes from
Kalshi's published fee schedule as I recall it and was NOT re-verified from the API: check it before trusting
any ROI figure. Resting (maker) orders are not modelled: we assume every trade crosses the spread.
"""
import math

TAKER = 0.07


def fee_per_contract(price, lots=None):
    """Fee in dollars per contract. lots=None: large-order limit (no cent rounding). lots=n: order of n contracts,
    rounded up to the cent as the exchange does (so a 1-lot order pays more per contract)."""
    raw = TAKER * price * (1 - price)
    if lots is None:
        return raw
    return math.ceil(round(raw * lots * 100, 6)) / 100 / lots
