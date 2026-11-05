"""Tiny polite HTTP helper (stdlib only): a pause between calls, retry on 429/5xx, JSON out."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "predmkt-research/0.1 (paper analysis; contact via GitHub repo)"
_last = [0.0]


def get(url, params=None, pause=0.35, tries=4, want_headers=False):
    """GET -> (status, parsed_json_or_text, headers). Never raises on HTTP errors."""
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    for k in range(tries):
        wait = pause - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body, status, hdr = r.read(), r.status, dict(r.headers)
        except urllib.error.HTTPError as e:
            body, status, hdr = e.read(), e.code, dict(e.headers)
        except Exception as e:  # network error
            if k == tries - 1:
                return 0, str(e), {}
            time.sleep(2 ** (k + 1))
            continue
        if status in (429, 500, 502, 503, 504) and k < tries - 1:
            time.sleep(2 ** (k + 2))
            continue
        text = body.decode("utf-8", "replace")
        try:
            return status, json.loads(text), hdr
        except ValueError:
            return status, text, hdr
    return 0, "retries exhausted", {}
