"""Turn bookmaker odds into fair probabilities and find positive-edge bets."""
from dataclasses import dataclass

OUTCOMES = ("H", "D", "A")


def devig(odds):
    """Remove the bookmaker margin (proportional method). odds: dict outcome->decimal odds."""
    inv = {k: 1 / v for k, v in odds.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def overround(odds):
    return sum(1 / v for v in odds.values()) - 1


def kelly_fraction(p, odds, fraction=0.25, cap=0.05):
    """Fractional Kelly stake as a share of bankroll, capped."""
    b = odds - 1
    f = (p * odds - 1) / b if b > 0 else 0.0
    return max(0.0, min(f * fraction, cap))


@dataclass
class Pick:
    outcome: str
    odds: float
    model_p: float
    market_p: float
    edge: float      # model_p - market_p (de-vigged)
    ev: float        # expected return per unit staked
    stake: float     # fraction of bankroll


def find_picks(model_probs, odds, min_ev=0.05, min_edge=0.03, max_odds=8.0, kelly=0.25):
    """Return Picks where the model sees value versus the (raw) offered odds."""
    market = devig(odds)
    picks = []
    for o in OUTCOMES:
        if o not in odds or odds[o] > max_odds:
            continue
        p, d = model_probs[o], odds[o]
        ev, edge = p * d - 1, p - market[o]
        if ev >= min_ev and edge >= min_edge:
            picks.append(Pick(o, d, p, market[o], edge, ev, kelly_fraction(p, d, kelly)))
    return sorted(picks, key=lambda x: -x.ev)
