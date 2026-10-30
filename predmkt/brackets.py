"""Market bracket geometry: which whole-degree highs make a Kalshi contract pay."""
import datetime as dt
import math
import re

MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def event_date(event_ticker):
    """'KXHIGHNY-26OCT29' -> date(2026,10,29). None if unparsable."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker or "")
    if not m or m.group(2) not in MON:
        return None
    try:
        return dt.date(2000 + int(m.group(1)), MON[m.group(2)], int(m.group(3)))
    except ValueError:
        return None


def bounds(strike_type, floor, cap):
    """Inclusive integer range [lo, hi] of daily highs for which YES pays. 'greater X' means >= X+1."""
    if strike_type == "between":
        return float(floor), float(cap)
    if strike_type == "greater":
        return float(floor) + 1, math.inf
    if strike_type == "less":
        return -math.inf, float(cap) - 1
    raise ValueError(f"unknown strike_type {strike_type!r}")


def label_bounds(subtitle):
    """Independent parse of the human label ('75° or above', '63° to 64°', '58° or below') for cross-checking."""
    s = (subtitle or "").replace("°", "")
    if m := re.match(r"\s*(-?\d+)\s+or above", s):
        return float(m.group(1)), math.inf
    if m := re.match(r"\s*(-?\d+)\s+or below", s):
        return -math.inf, float(m.group(1))
    if m := re.match(r"\s*(-?\d+)\s+to\s+(-?\d+)", s):
        return float(m.group(1)), float(m.group(2))
    return None


def pays(lo, hi, high):
    return lo <= high <= hi
