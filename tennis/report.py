"""Assemble tennis_report.txt: data checks, sanity checks, forecast quality, calibration, betting, slices, controls."""
import math
import re

import numpy as np
import pandas as pd

from . import data, synth
from .fit import slope_with_se, logit, sigmoid, fit_logit
from .market import Z, log_loss, mean_moe, paired_gap, pm, reliability, settle, summarise
from .pipeline import MIN_PRIOR, forecast_frame, test_sample

THRESHOLDS = (0.0, 0.02, 0.05, 0.10)
SLICE_THR = 0.05  # the one betting rule used for the slice tables, fixed before looking at any result
TEST_START = "2013-01-01"
LEVEL_ORDER = ["Grand Slam", "Masters 1000", "500", "250", "Finals", "other"]


def tbl(df, **kw):
    return df.to_string(index=False, **kw)


def z_crit(alpha):
    """Two-sided normal critical value for tail probability alpha (bisection; no scipy)."""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------ data section

def data_section(matches, diags, f, s):
    L = ["1. DATA AND CLEANING", "=" * 60]
    ok = [d for d in diags if "error" not in d]
    bad = [d for d in diags if "error" in d]
    for tour in ("ATP", "WTA"):
        dd = [d for d in ok if d["tour"] == tour]
        if not dd:
            L.append(f"{tour}: no files")
            continue
        st = {}
        for d in dd:
            for k, v in d["status"].items():
                st[k] = st.get(k, 0) + v
        L.append(f"{tour}: {len(dd)} files, years {min(d['year'] for d in dd)}-{max(d['year'] for d in dd)}, "
                 f"{sum(d['rows'] for d in dd)} rows read, {sum(d['kept'] for d in dd)} finished matches kept")
        L.append(f"   Comment column values (all rows): {dict(sorted(st.items(), key=lambda kv: -kv[1]))}")
        L.append(f"   dropped as unfinished (retired/walkover/disqualified/etc.): {sum(d['dropped_unfinished'] for d in dd)}; "
                 f"of which 'completed' but winner short of the sets needed: {sum(d['unfinished_by_sets'] for d in dd)}; "
                 f"bad rows (no date/name): {sum(d['bad_rows'] for d in dd)}")
        nocomment = [d["year"] for d in dd if not d["comment_column_used"]]
        if nocomment:
            L.append(f"   years with no usable Comment column (filtered on set counts only): {nocomment}")
    for d in bad:
        L.append(f"UNREADABLE: {d['file']}: {d['error']}")
    cols = set()
    for d in ok:
        cols |= set(d["columns"])
    close_cols = sorted(c for c in cols if re.fullmatch(r"(PS|B365|Avg|Max|CB|GB|IW|SB|EX|UB|LB|SJ)C[WL]", c))
    open_cols = sorted(c for c in cols if re.fullmatch(r"(PS|B365|Avg|Max|CB|GB|IW|SB|EX|UB|LB|SJ)[WL]", c))
    L.append(f"price columns in the files: {open_cols}")
    L.append(f"separate closing-price columns: {close_cols if close_cols else 'NONE'}")
    L.append("")
    L.append("Decisions (fixed before looking at results):")
    L.append(" - Retirements, walkovers, disqualifications and any match where the winner has too few sets are removed")
    L.append("   from BOTH model training and the betting tests. (Real books void or settle these inconsistently. Removing")
    L.append("   them uses hindsight, but it removes the same matches for the model and the market, so comparisons stay fair;")
    L.append("   it does flatter any betting ROI slightly because a real bettor cannot know in advance a match will be finished.)")
    L.append(f" - Player order is randomised by a fingerprint of (pair of names, date, tournament, round), never by who won.")
    L.append(f" - Elo is built from all matches from the first file on. Evaluation starts {TEST_START}; matches where either player")
    L.append(f"   has fewer than {MIN_PRIOR} earlier matches are left out (no usable rating). A robustness line without that filter is shown.")
    L.append(" - Surface 'Carpet' is treated as hard. Tournament levels come from the Series/Tier text; 'International Gold' is counted as 500,")
    L.append("   WTA 'Premier' as 500, 'Premier Mandatory' / 'Premier 5' / WTA1000 as Masters 1000. This mapping is approximate.")
    L.append(" - Players are matched by name text (e.g. 'Djokovic N.'). Two players sharing surname+initial would be merged; renamed spellings split.")
    L.append("")
    L.append("Price availability by year (share of kept matches with each price; both players' prices needed):")
    y = f.assign(year=f["date"].dt.year)
    cov = y.groupby("year").agg(matches=("y", "size"), pinnacle=("ps_o1", lambda x: x.notna().mean()),
                                average=("avg_o1", lambda x: x.notna().mean()), best=("max_o1", lambda x: x.notna().mean()),
                                bet365=("b365_o1", lambda x: x.notna().mean()), close=("psc_o1", lambda x: x.notna().mean()))
    for c in cov.columns[1:]:
        cov[c] = (cov[c] * 100).round(0).astype(int).astype(str) + "%"
    L.append(tbl(cov.reset_index()))
    L.append("")
    L.append(f"Evaluation sample (>= {TEST_START}, warm players, Pinnacle+average+best prices all present, model forecasts present): {len(s)} matches")
    L.append("   " + ", ".join(f"{t}: {n}" for t, n in s["tour"].value_counts().items()))
    L.append("   by level: " + ", ".join(f"{k}: {v}" for k, v in s["level"].value_counts().items()))
    L.append("   by surface: " + ", ".join(f"{k}: {v}" for k, v in s["surface"].value_counts().items()))
    unk = int((f["rround"] == 0).sum())
    L.append(f"   matches whose Round text was not recognised (processed first on their date): {unk}")
    return L


# ------------------------------------------------------------ sanity

def sanity_section(f, s):
    L = ["", "2. SANITY CHECKS (does the model reproduce what the data itself says?)", "=" * 60]
    L.append(f"Share of matches where the randomised 'player 1' won: {f['y'].mean():.4f} over {len(f)} matches "
             f"(should be 0.5; ±{Z * 0.5 / math.sqrt(len(f)):.4f} is chance)")
    L.append(f"Evaluation sample: mean y {s['y'].mean():.4f} vs mean forecast  Elo {s['pA'].mean():.4f}  stack {s['pB'].mean():.4f}  Pinnacle {s['ps_p1'].mean():.4f}")
    w = f[f["warm"] & f["pA"].notna()]
    L.append(f"All warm matches since the start (Elo is online, so every prediction is out-of-sample): mean forecast {w['pA'].mean():.4f} vs observed {w['y'].mean():.4f}")
    r = s[(s["r1"].notna()) & (s["r2"].notna()) & (s["r1"] != s["r2"])]
    better = (r["r1"] < r["r2"]).to_numpy()
    obs = np.where(better, r["y"], 1 - r["y"])
    L.append(f"Better-ranked player wins: observed {obs.mean():.4f} (±{Z * obs.std(ddof=1) / math.sqrt(len(obs)):.4f}, n={len(obs)}) vs forecast: "
             + "  ".join(f"{n} {np.where(better, r[c], 1 - r[c]).mean():.4f}" for n, c in (("Elo", "pA"), ("stack", "pB"), ("Pinnacle", "ps_p1"), ("average", "avg_p1"))))
    L.append("Calibration slope (1.00 = well calibrated; <1 overconfident; >1 underconfident; ± is 1 standard error):")
    for n, c in (("Elo only (A)", "pA"), ("Elo+rank stack (B)", "pB"), ("market-anchored (C)", "pC"), ("Pinnacle", "ps_p1"), ("average", "avg_p1")):
        if s[c].notna().any():
            b, se = slope_with_se(s[c].to_numpy(), s["y"].to_numpy())
            L.append(f"   {n:<22} slope {b:.3f} ±{se:.3f}")
    th = f.attrs.get("theta_B")
    if th is not None:
        L.append(f"Stack B final coefficients [elo logit, log rank ratio, log points ratio]: {np.round(th, 3).tolist()}")
        L.append("   (Elo ratings are noisy, so a coefficient of about 0.5-1.0 on the Elo logit is normal. Near 0, negative or far above 1 would point to a scaling bug.)")
    thc = f.attrs.get("theta_C")
    if thc is not None:
        L.append(f"Anchored C final coefficients [elo disagreement with Pinnacle, Pinnacle tilt]: {np.round(thc, 3).tolist()} (0 = just copy Pinnacle)")
    # in-sample vs out-of-sample for B
    from .elo import LN10_400
    from .pipeline import _logratio
    X = np.column_stack([f["elo_diff"] * LN10_400, _logratio(f["r1"], f["r2"]), _logratio(f["pts2"], f["pts1"])])
    tr = f["warm"].to_numpy() & (f["date"] < pd.Timestamp(TEST_START)).to_numpy() & np.isfinite(X).all(1) & f["ps_p1"].notna().to_numpy()
    if th is not None and tr.sum() > 500:
        p_in, yy = sigmoid(X[tr] @ th), f["y"].to_numpy()[tr]
        L.append(f"Stack B on the {tr.sum()} pre-{TEST_START[:4]} matches that have Pinnacle prices, using the final coefficients (in-sample): log loss {log_loss(p_in, yy).mean():.4f} "
                 f"(Pinnacle on the same matches {log_loss(f['ps_p1'].to_numpy()[tr], yy).mean():.4f}); mean forecast {p_in.mean():.4f} vs observed {yy.mean():.4f}.")
        L.append(f"   Out-of-sample walk-forward: B {log_loss(s['pB'], s['y']).mean():.4f} vs Pinnacle {log_loss(s['ps_p1'], s['y']).mean():.4f}. "
                 "(Different match sets, so only the gaps are comparable: a model that is much better in-sample than out-of-sample relative to Pinnacle is overfit.)")
    L.append(f"Coin-flip log loss for reference: {math.log(2):.4f}")
    return L


# ------------------------------------------------------------ forecast quality

MODELS = (("Elo only (A)", "pA"), ("Elo+rank+points stack (B)", "pB"), ("Market-anchored (C)", "pC"),
          ("Pinnacle fair", "ps_p1"), ("Average fair", "avg_p1"), ("Bet365 fair", "b365_p1"))


def forecast_table(s):
    y = s["y"].to_numpy()
    base = {k: log_loss(s[c], y) for k, c in (("ps", "ps_p1"), ("avg", "avg_p1"))}
    rows = []
    for name, c in MODELS:
        if c not in s or s[c].isna().all():
            continue
        ok = s[c].notna().to_numpy()
        ll = log_loss(s[c].to_numpy(), y)
        n, m, moe = mean_moe(ll[ok])
        row = {"forecast": name, "n": n, "log loss": f"{m:.4f} ±{moe:.4f}",
               "accuracy": f"{((s[c].to_numpy()[ok] >= 0.5) == (y[ok] == 1)).mean() * 100:.1f}%"}
        for k, lab in (("ps", "vs Pinnacle"), ("avg", "vs average")):
            _, g, gm = paired_gap(ll[ok], base[k][ok])
            row[lab] = "-" if c in ("ps_p1",) and k == "ps" else f"{g:+.4f} ±{gm:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_section(s):
    L = ["", "3. FORECAST QUALITY (out-of-sample log loss; lower is better)", "=" * 60,
         "'vs Pinnacle' is the paired difference (model minus Pinnacle) over the same matches, with a 95% margin.",
         "Negative and clear of zero would mean the model beats the market. Positive means the market is better."]
    for lab, sub in (("ALL", s), ("ATP", s[s["tour"] == "ATP"]), ("WTA", s[s["tour"] == "WTA"])):
        if len(sub) >= 200:
            L += ["", f"--- {lab} ({len(sub)} matches)", tbl(forecast_table(sub))]
    return L


def calibration_section(s):
    L = ["", "4. CALIBRATION BY PROBABILITY BAND (each match's favourite side, so each match counts once)", "=" * 60,
         "gap = predicted - observed (positive = overconfident). moe = 95% margin on the observed rate."]
    for name, c in (("Stack B", "pB"), ("Pinnacle fair", "ps_p1")):
        t = reliability(s[c].to_numpy(), s["y"].to_numpy())
        t[["predicted", "observed", "gap", "moe"]] = t[["predicted", "observed", "gap", "moe"]].round(3)
        L += ["", f"--- {name}", tbl(t)]
    return L


# ------------------------------------------------------------ betting

def strategies(s):
    return [
        ("Model B vs Pinnacle price", "pB", "ps"),
        ("Model B vs average price", "pB", "avg"),
        ("Model B vs best price", "pB", "max"),
        ("Elo only (A) vs best price", "pA", "max"),
        ("Market-anchored (C) vs best price", "pC", "max"),
        ("NO MODEL: Pinnacle fair price vs best price", "ps_p1", "max"),
    ]


def betting_section(s, f):
    L = ["", "5. BETTING SIMULATION (flat 1-unit stakes, paper only)", "=" * 60,
         "Each match: take the side with the higher expected value (model probability x price - 1); bet if it clears the threshold.",
         "ROI is profit per unit staked in %, with a 95% margin of error. h1/h2 = first/second half of the test period (by date).",
         "Prices above 15.0 are ignored. The 'best price' is the highest across the listed bookmakers and assumes you hold accounts",
         "everywhere and get on at that price with no limits - an upper bound, not a realistic result."]
    late = (s["date"] >= s["date"].median()).to_numpy()
    close = s["psc_p1"].to_numpy() if s["psc_p1"].notna().mean() > 0.5 else None
    L.append(f"Closing-line value (CLV): {'measured against separate closing prices' if close is not None else 'CANNOT be measured - the files give one price snapshot per bookmaker, no separate opening and closing price'}.")
    rows = []
    for name, pc, src in strategies(s):
        if s[pc].isna().all():
            continue
        for thr in THRESHOLDS:
            bet, profit, odds, clv = settle(s[pc], s[f"{src}_o1"], s[f"{src}_o2"], s["y"], thr, close)
            r = summarise(bet, profit, odds, clv, late)
            rows.append({"strategy": name, "min EV": f"{thr:.0%}", "bets": r["bets"],
                         "hit": f"{r['hit'] * 100:.1f}%" if r["bets"] else "-", "avg odds": f"{r['odds']:.2f}" if r["bets"] else "-",
                         "ROI %": pm(r.get("roi"), r.get("moe")), "ROI h1": pm(r.get("h1"), r.get("h1_moe")),
                         "ROI h2": pm(r.get("h2"), r.get("h2_moe")), **({"CLV %": pm(r.get("clv"), None)} if close is not None else {})})
    for name, src in (("BLIND: always player 1 at best price", "max"), ("BLIND: always player 1 at Pinnacle price", "ps")):
        profit = np.where(s["y"] == 1, s[f"{src}_o1"] - 1, -1.0)
        r = summarise(np.ones(len(s), bool), profit, s[f"{src}_o1"].to_numpy(), np.full(len(s), np.nan), late)
        rows.append({"strategy": name, "min EV": "-", "bets": r["bets"], "hit": f"{r['hit'] * 100:.1f}%", "avg odds": f"{r['odds']:.2f}",
                     "ROI %": pm(r["roi"], r["moe"]), "ROI h1": pm(r["h1"], r["h1_moe"]), "ROI h2": pm(r["h2"], r["h2_moe"])})
    L += ["", tbl(pd.DataFrame(rows).fillna("-"))]
    # robustness: same headline rule without the warm-player filter
    allm = f[(f["date"] >= pd.Timestamp(TEST_START)) & f["pB"].notna() & f["ps_o1"].notna() & f["max_o1"].notna()]
    if len(allm):
        late2 = (allm["date"] >= allm["date"].median()).to_numpy()
        bet, profit, odds, clv = settle(allm["pB"], allm["max_o1"], allm["max_o2"], allm["y"], SLICE_THR)
        r = summarise(bet, profit, odds, clv, late2)
    L += ["", f"Robustness - Model B vs best price at {SLICE_THR:.0%} EV, WITHOUT dropping unrated players ({len(allm)} matches): "
          + (f"{r['bets']} bets, ROI {pm(r.get('roi'), r.get('moe'))}%" if allm.shape[0] and r["bets"] else "no bets")]
    return L


# ------------------------------------------------------------ slices

def rank_band(s):
    best = np.fmin(s["r1"].to_numpy(float), s["r2"].to_numpy(float))
    out = pd.cut(best, [0, 10, 30, 100, 200, 1e9], labels=["1-10", "11-30", "31-100", "101-200", "201+"])
    return pd.Series(out.astype(object), index=s.index).fillna("unranked")


def slice_section(s):
    L = ["", "6. WHERE MIGHT PRICES BE SOFTER? (slices; model B vs best price, rule fixed in advance: EV >= "
         f"{SLICE_THR:.0%})", "=" * 60]
    fav = np.maximum(s["ps_p1"], 1 - s["ps_p1"])
    dims = {
        "tour": s["tour"], "level": s["level"], "surface": s["surface"], "round": s["round"],
        "rank band (better player)": rank_band(s),
        "Pinnacle favourite win prob": pd.cut(fav, [0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True).astype(str),
        "year": s["date"].dt.year.astype(str),
    }
    bet, profit, odds, clv = settle(s["pB"], s["max_o1"], s["max_o2"], s["y"], SLICE_THR)
    llm = log_loss(s["pB"].to_numpy(), s["y"].to_numpy())
    llp = log_loss(s["ps_p1"].to_numpy(), s["y"].to_numpy())
    ovr_ps = (1 / s["ps_o1"] + 1 / s["ps_o2"] - 1).to_numpy()
    ovr_avg = (1 / s["avg_o1"] + 1 / s["avg_o2"] - 1).to_numpy()
    rows = []
    for dim, ser in dims.items():
        for val, idx in pd.Series(np.arange(len(s))).groupby(ser.to_numpy()):
            i = idx.to_numpy()
            if len(i) < 30:
                continue
            _, g, gm = paired_gap(llm[i], llp[i])
            b = bet[i]
            nb = int(b.sum())
            if nb > 1:
                sd = profit[i][b].std(ddof=1)
                roi, moe = profit[i][b].mean(), Z * sd / math.sqrt(nb)
            else:
                roi = moe = np.nan
            rows.append({"dimension": dim, "slice": val, "matches": len(i), "margin PS %": f"{ovr_ps[i].mean() * 100:.1f}",
                         "margin avg %": f"{ovr_avg[i].mean() * 100:.1f}", "B minus PS logloss": f"{g:+.4f} ±{gm:.4f}",
                         "bets": nb, "ROI %": pm(roi, moe), "_z": roi / (moe / Z) if nb > 1 and moe > 0 else np.nan, "_nb": nb})
    t = pd.DataFrame(rows)
    shown = t.drop(columns=["_z", "_nb"]).copy()
    L += ["margin = bookmaker overround (extra % over 100 baked into the two prices). 'B minus PS logloss' < 0 would mean B forecasts better than Pinnacle there.",
          "Slices with fewer than 30 matches are omitted. Every number carries a 95% margin of error; many of these are huge.", "", tbl(shown)]
    big = t[t["_nb"] >= 100]
    m = len(big)
    pos = int((big["_z"] > Z).sum())
    neg = int((big["_z"] < -Z).sum())
    L += ["", "MULTIPLE TESTING WARNING: this table is "
          f"{len(t)} slices (overlapping - the same match appears in a tour, a level, a surface, a round, a rank band, a price band and a year).",
          f"Among the {m} slices with at least 100 bets, {pos} look profitable at the 95% level and {neg} look unprofitable. Pure chance alone would give",
          f"about {0.025 * m:.1f} 'profitable' and {0.025 * m:.1f} 'unprofitable' ones. After a Bonferroni correction for {m} tests a slice would need z > {z_crit(0.05 / max(m, 1)):.2f}"
          f" (best z seen: {big['_z'].max() if m else float('nan'):.2f}; worst: {big['_z'].min() if m else float('nan'):.2f}).",
          "Treat any single 'good-looking' slice as a hypothesis to test on NEW data, not as a finding."]
    return L


# ------------------------------------------------------------ controls

def controls_section():
    L = ["", "7. MACHINERY CONTROLS (synthetic matches with a KNOWN truth - not real tennis)", "=" * 60,
         "Same code, fake data. If the pipeline is sound it should find nothing when prices are efficient and find an edge when prices are noisy."]
    for label, sd in (("NEGATIVE control: prices = true probabilities + margin", 0.0), ("POSITIVE control: prices blurred by random error (sd 0.4 on the log-odds scale)", 0.4)):
        m, _ = data.clean(synth.synth_raw(noise_sd=sd, seed=3), "ATP")
        f = forecast_frame(data.orient(m), test_start="2010-01-01")
        s = test_sample(f, "2010-01-01")
        _, g, gm = paired_gap(log_loss(s["pB"], s["y"]), log_loss(s["ps_p1"], s["y"]))
        _, gc, gcm = paired_gap(log_loss(s["pC"], s["y"]), log_loss(s["ps_p1"], s["y"]))
        bet, profit, odds, clv = settle(s["pB"], s["max_o1"], s["max_o2"], s["y"], SLICE_THR)
        r = summarise(bet, profit, odds, clv, np.zeros(len(s), bool))
        L.append(f"{label}: {len(s)} matches. Log loss B minus Pinnacle {g:+.4f} ±{gm:.4f}; C minus Pinnacle {gc:+.4f} ±{gcm:.4f}; "
                 f"B vs best price at {SLICE_THR:.0%} EV: {r['bets']} bets, ROI {pm(r.get('roi'), r.get('moe'))}%")
    return L


# ------------------------------------------------------------ headline

def headline(s):
    L = ["", "8. HEADLINE (generated by rule from the numbers above; read the sections for context)", "=" * 60]
    y = s["y"].to_numpy()
    llp = log_loss(s["ps_p1"], y)
    for name, c in (("B (Elo+rank stack)", "pB"), ("C (market-anchored)", "pC")):
        _, g, gm = paired_gap(log_loss(s[c], y), llp)
        verdict = "model BETTER than Pinnacle" if g + gm < 0 else "Pinnacle BETTER than model" if g - gm > 0 else "no clear difference"
        L.append(f"Forecasts, {name} vs Pinnacle: log loss gap {g:+.4f} ±{gm:.4f} -> {verdict}.")
    late = (s["date"] >= s["date"].median()).to_numpy()
    for name, pc, src in strategies(s)[:5]:
        if s[pc].isna().all():
            continue
        bet, profit, odds, clv = settle(s[pc], s[f"{src}_o1"], s[f"{src}_o2"], y, SLICE_THR)
        r = summarise(bet, profit, odds, clv, late)
        if not r["bets"]:
            continue
        good = r["roi"] - r["moe"] > 0 and r.get("h1", -1) > 0 and r.get("h2", -1) > 0
        L.append(f"Betting, {name}, EV>={SLICE_THR:.0%}: {r['bets']} bets, ROI {pm(r['roi'], r['moe'])}% -> "
                 + ("profit clear of zero in both halves" if good else "no reliable profit" if r["roi"] - r["moe"] <= 0 else "positive overall but NOT in both halves"))
    return L


def build_report(matches, diags, test_start=TEST_START, with_controls=True):
    o = data.orient(matches)
    f = forecast_frame(o, test_start=test_start)
    s = test_sample(f, test_start)
    if len(s) < 500:
        raise SystemExit(f"only {len(s)} evaluation matches - check price coverage")
    L = ["TENNIS BACKTEST: can a simple model built on free data beat bookmaker prices?",
         "Paper analysis only. Data: tennis-data.co.uk. Walk-forward: every match is forecast using only earlier matches.", ""]
    for part in (data_section(matches, diags, f, s), sanity_section(f, s), forecast_section(s), calibration_section(s),
                 betting_section(s, f), slice_section(s), controls_section() if with_controls else [], headline(s)):
        L += part
    return "\n".join(L) + "\n"
