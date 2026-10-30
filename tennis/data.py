"""Fetch, clean and orient the tennis-data.co.uk yearly files (ATP, and WTA with a `w` folder).

Two traps are handled here on purpose:
  * the files list "Winner" and "Loser", so a model that sees who is listed first knows the answer.
    `orient` re-labels every match as player 1 / player 2 with a coin flip that depends only on the
    unordered pair of names, date, tournament and round, never on who won.
  * retirements, walkovers and other unfinished matches are dropped (a bet on those is not a normal result).
"""
import hashlib
import io
import re
import sys
import time
import zipfile
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE = "http://www.tennis-data.co.uk/"
INDEX_PAGES = ("alldata.php", "data.php")
FIRST_YEAR = {"atp": 2000, "wta": 2007}
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# price source -> (winner column, loser column). Optional "psc" is only used if the file has closing prices.
ODDS = {
    "ps": ("PSW", "PSL"),        # Pinnacle
    "avg": ("AvgW", "AvgL"),     # average across bookmakers
    "max": ("MaxW", "MaxL"),     # best price on offer
    "b365": ("B365W", "B365L"),
    "psc": ("PSCW", "PSCL"),     # Pinnacle closing (probably absent)
}
ROUND_RANK = {"1st round": 1, "2nd round": 2, "3rd round": 3, "4th round": 4, "round robin": 3,
              "quarterfinals": 5, "semifinals": 6, "the final": 7}
SURFACES = ("hard", "clay", "grass")


# ---------------------------------------------------------------- download

def http_get(url, retries=3, timeout=60):
    """Bytes of the URL (redirects are followed by urllib), or None if it is missing / unreachable."""
    last = None
    for i in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return r.read()
        except HTTPError as e:
            if e.code in (404, 410):
                return None
            last = e
        except (URLError, OSError) as e:
            last = e
        time.sleep(2 ** i)
    print(f"  giving up on {url}: {last}", file=sys.stderr)
    return None


def looks_like_workbook(b):
    """xlsx and zip start with PK; old xls is an OLE2 file. Anything else (an HTML error page) is not data."""
    return bool(b) and (b[:2] == b"PK" or b[:4] == b"\xd0\xcf\x11\xe0")


LINK = re.compile(r"""href=["']([^"']+?\.(?:xlsx|xls|zip))["']""", re.I)
EXT_ORDER = {".xlsx": 0, ".xls": 1, ".zip": 2}


def parse_links(html, page_url=BASE):
    """{(tour, year): [urls, best format first]} from a data page's links."""
    found = {}
    for href in LINK.findall(html):
        url = urljoin(page_url, href)
        m = re.search(r"/(\d{4})(w?)/[^/]+$", url)
        if not m:
            continue
        key = ("wta" if m.group(2) else "atp", int(m.group(1)))
        if url not in found.setdefault(key, []):
            found[key].append(url)
    for urls in found.values():
        urls.sort(key=lambda u: EXT_ORDER.get(Path(u).suffix.lower(), 9))
    return found


def guess_urls(tour, year):
    folder = f"{year}{'w' if tour == 'wta' else ''}"
    return [f"{BASE}{folder}/{year}{ext}" for ext in (".xlsx", ".xls", ".zip")]


def fetch_all(out_dir, last_year=None, log=print):
    """Download every available yearly file. Returns a manifest list of dicts. Existing past years are kept."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_year = last_year or date.today().year
    discovered = {}
    for page in INDEX_PAGES:
        raw = http_get(BASE + page)
        if raw:
            for k, v in parse_links(raw.decode("latin-1"), BASE + page).items():
                discovered.setdefault(k, [])
                discovered[k] += [u for u in v if u not in discovered[k]]
    log(f"index pages listed {len(discovered)} yearly files")
    notes = http_get(BASE + "notes.txt")
    if notes:
        (out / "notes.txt").write_bytes(notes)
    manifest = []
    for tour, first in FIRST_YEAR.items():
        for year in range(first, last_year + 1):
            have = sorted(out.glob(f"{tour}_{year}.*"))
            if have and year < last_year:
                manifest.append({"tour": tour, "year": year, "file": have[0].name, "url": "cached"})
                continue
            urls = discovered.get((tour, year), []) + [u for u in guess_urls(tour, year) if u not in discovered.get((tour, year), [])]
            for url in urls:
                b = http_get(url)
                if looks_like_workbook(b):
                    name = f"{tour}_{year}{Path(url).suffix.lower()}"
                    (out / name).write_bytes(b)
                    manifest.append({"tour": tour, "year": year, "file": name, "url": url})
                    break
            else:
                log(f"  no file found for {tour} {year}")
    return manifest


# ---------------------------------------------------------------- read + clean

def read_workbook(path):
    """One yearly file -> DataFrame (xlsx via openpyxl, xls via xlrd, or a zip holding either)."""
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls", ".csv"))]
            if not names:
                raise ValueError(f"{p.name}: zip holds no spreadsheet")
            raw = z.read(sorted(names, key=lambda n: EXT_ORDER.get(Path(n).suffix.lower(), 9))[0])
    try:
        return pd.read_excel(io.BytesIO(raw), sheet_name=0)
    except Exception as e:  # last resort: some very old files are really CSV
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        except Exception:
            raise ValueError(f"{p.name}: cannot read as a workbook ({e})")


def level_of(series):
    s = str(series).strip().lower()
    if "grand slam" in s:
        return "Grand Slam"
    if "masters cup" in s or "tour championship" in s or "finals" in s:
        return "Finals"
    if "masters" in s or "1000" in s or "premier mandatory" in s or "premier 5" in s:
        return "Masters 1000"
    if "500" in s or "international gold" in s or s == "premier":
        return "500"
    if "250" in s or "international" in s:
        return "250"
    return "other"


def clean(raw, tour):
    """Standardise one yearly file. Returns (matches, diagnostics). `matches` still has winner/loser roles."""
    d = raw.copy()
    d.columns = [str(c).strip() for c in d.columns]
    d = d.rename(columns={"Tier": "Series"})  # WTA files call it Tier
    for need in ("Date", "Winner", "Loser"):
        if need not in d:
            raise ValueError(f"missing column {need}; have {list(d.columns)[:20]}")
    for opt in ("Tournament", "Series", "Surface", "Round", "Best of", "Comment", "WRank", "LRank", "WPts", "LPts", "Wsets", "Lsets"):
        if opt not in d:
            d[opt] = np.nan
    diag = {"rows": len(d)}
    d["date"] = pd.to_datetime(d["Date"], errors="coerce", dayfirst=True)
    d["winner"] = d["Winner"].astype("string").str.strip()
    d["loser"] = d["Loser"].astype("string").str.strip()
    ok = d["date"].notna() & d["winner"].notna() & d["loser"].notna() & (d["winner"] != d["loser"])
    diag["bad_rows"] = int((~ok).sum())
    d = d[ok].copy()

    comment = d["Comment"].astype("string").str.strip().str.lower()
    informative = bool(comment.notna().any())
    status = comment.fillna("(blank)") if informative else pd.Series("completed", index=d.index)
    sets_w, sets_l = pd.to_numeric(d["Wsets"], errors="coerce"), pd.to_numeric(d["Lsets"], errors="coerce")
    best_of = pd.to_numeric(d["Best of"], errors="coerce").fillna(3)
    short = sets_w.notna() & (sets_w < best_of // 2 + 1)  # winner did not win enough sets: unfinished
    keep = (status == "completed") & ~short
    diag["status"] = status.value_counts().to_dict()
    diag["comment_column_used"] = informative
    diag["unfinished_by_sets"] = int((short & (status == "completed")).sum())
    diag["dropped_unfinished"] = int((~keep).sum())
    d = d[keep].copy()

    out = pd.DataFrame({
        "tour": tour,
        "date": d["date"],
        "tourney": d["Tournament"].astype("string").str.strip().fillna(""),
        "level": d["Series"].map(level_of),
        "surface": d["Surface"].astype("string").str.strip().str.lower().replace({"carpet": "hard"}).fillna("unknown"),
        "round": d["Round"].astype("string").str.strip().fillna(""),
        "best_of": best_of[d.index],
        "winner": d["winner"], "loser": d["loser"],
        "wrank": pd.to_numeric(d["WRank"], errors="coerce"), "lrank": pd.to_numeric(d["LRank"], errors="coerce"),
        "wpts": pd.to_numeric(d["WPts"], errors="coerce"), "lpts": pd.to_numeric(d["LPts"], errors="coerce"),
    })
    out["rround"] = out["round"].str.lower().map(ROUND_RANK).fillna(0).astype(int)
    out.loc[~out["surface"].isin(SURFACES), "surface"] = "unknown"
    for src, (w, l) in ODDS.items():
        ow = pd.to_numeric(d[w], errors="coerce") if w in d else pd.Series(np.nan, index=d.index)
        ol = pd.to_numeric(d[l], errors="coerce") if l in d else pd.Series(np.nan, index=d.index)
        good = (ow > 1) & (ol > 1)
        out[f"{src}_w"], out[f"{src}_l"] = ow.where(good), ol.where(good)
    diag["columns"] = list(d.columns)
    return out.reset_index(drop=True), diag


def load_dir(data_dir, log=print):
    """Read every downloaded file. Returns (all matches, per-file diagnostics)."""
    frames, diags = [], []
    for p in sorted(Path(data_dir).glob("*_*.*")):
        m = re.match(r"(atp|wta)_(\d{4})\.(xlsx|xls|zip)$", p.name)
        if not m:
            continue
        try:
            df, diag = clean(read_workbook(p), m.group(1).upper())
        except Exception as e:
            log(f"  could not use {p.name}: {e}")
            diags.append({"file": p.name, "error": str(e)})
            continue
        diag.update(file=p.name, tour=m.group(1).upper(), year=int(m.group(2)), kept=len(df))
        frames.append(df)
        diags.append(diag)
    if not frames:
        raise SystemExit("no data files could be read")
    return pd.concat(frames, ignore_index=True), diags


# ---------------------------------------------------------------- orientation (the leak fix)

def _flip_bit(seed, tour, day, tourney, rnd, a, b):
    lo, hi = sorted((a, b))
    h = hashlib.md5(f"{seed}|{tour}|{day}|{tourney}|{rnd}|{lo}|{hi}".encode()).hexdigest()
    return int(h[:8], 16) & 1, lo, hi


def orient(m, seed=7):
    """Re-label winner/loser as player 1 / player 2 so that the label carries no information about the result.

    The coin flip depends only on the unordered pair of names plus date, tournament and round, so it is the same
    whichever player the file lists as winner. y = 1 when player 1 won. Odds are re-pointed to player 1 / 2.
    """
    n = len(m)
    flips, lo, hi = np.empty(n, int), np.empty(n, object), np.empty(n, object)
    for i, (t, d, tn, rd, w, l) in enumerate(zip(m["tour"], m["date"].dt.strftime("%Y-%m-%d"), m["tourney"], m["round"], m["winner"], m["loser"])):
        flips[i], lo[i], hi[i] = _flip_bit(seed, t, d, tn, rd, w, l)
    p1 = np.where(flips == 0, lo, hi)
    win = m["winner"].to_numpy(object)
    y = (p1 == win).astype(int)
    p1_is_w = y == 1
    o = m.drop(columns=["winner", "loser", "wrank", "lrank", "wpts", "lpts"]).copy()
    o["p1"] = p1
    o["p2"] = np.where(p1_is_w, m["loser"].to_numpy(object), win)
    o["y"] = y
    for a, b, name in (("wrank", "lrank", "r"), ("wpts", "lpts", "pts")):
        o[f"{name}1"] = np.where(p1_is_w, m[a], m[b])
        o[f"{name}2"] = np.where(p1_is_w, m[b], m[a])
    for src in ODDS:
        o = o.drop(columns=[f"{src}_w", f"{src}_l"])
        o[f"{src}_o1"] = np.where(p1_is_w, m[f"{src}_w"], m[f"{src}_l"])
        o[f"{src}_o2"] = np.where(p1_is_w, m[f"{src}_l"], m[f"{src}_w"])
    return o.sort_values(["date", "rround"], kind="stable").reset_index(drop=True)
