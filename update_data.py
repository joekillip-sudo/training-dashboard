import os
from dotenv import load_dotenv
load_dotenv()
import json
from datetime import datetime, timezone
from fredapi import Fred
import feedparser

fred = Fred(api_key=os.environ["FRED_API_KEY"])

def series_history(series_id, points):
    s = fred.get_series(series_id).dropna().tail(points)
    return [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)} for d, v in s.items()]

import pandas as pd

def yoy_history(series_id, points):
    s = fred.get_series(series_id)
    s.index = s.index.to_period("M")
    full_range = pd.period_range(s.index.min(), s.index.max(), freq="M")
    s = s.reindex(full_range)  # fills any missing months with NaN, preserving true calendar spacing
    yoy = ((s / s.shift(12)) - 1) * 100
    yoy = yoy.dropna().tail(points)
    return [{"date": d.strftime("%Y-%m-01"), "value": round(float(v), 2)} for d, v in yoy.items()]

indicators = {}

indicators["us_10y"] = {"label": "US 10Y Treasury Yield", "unit": "%", "history": series_history("DGS10", 30)}
indicators["fed_rate"] = {"label": "Fed Funds Rate", "unit": "%", "history": series_history("FEDFUNDS", 12)}
indicators["us_cpi_yoy"] = {"label": "US Inflation (YoY)", "unit": "%", "history": yoy_history("CPIAUCSL", 12)}
indicators["us_debt_gdp"] = {"label": "US Federal Debt (% of GDP)", "unit": "%", "history": series_history("GFDEGDQ188S", 12)}
indicators["gdp_growth"] = {"label": "US GDP Growth (QoQ annualized)", "unit": "%", "history": series_history("A191RL1Q225SBEA", 8)}
indicators["sp500"] = {"label": "S&P 500", "unit": "", "history": series_history("SP500", 30)}
indicators["dollar_index"] = {"label": "US Dollar Index", "unit": "", "history": series_history("DTWEXBGS", 30)}
indicators["oil_wti"] = {"label": "Crude Oil (WTI)", "unit": " $/bbl", "history": series_history("DCOILWTICO", 30)}

# For every indicator, derive current value, trend, and change from its OWN history —
# this is the fix for the earlier bug: comparisons are always against the last real
# published data point, never just "whatever the last hourly run happened to see."
for key, info in indicators.items():
    hist = info["history"]
    if len(hist) == 0:
        info["value"] = None
        info["trend"] = "same"
        info["change"] = None
        info["compared_date"] = None
        continue

    info["value"] = hist[-1]["value"]

    if len(hist) >= 2:
        previous = hist[-2]["value"]
        change = round(info["value"] - previous, 2)
        info["change"] = change
        info["compared_date"] = hist[-2]["date"]
        info["trend"] = "up" if change > 0 else ("down" if change < 0 else "same")
    else:
        info["change"] = None
        info["compared_date"] = None
        info["trend"] = "same"

def fetch_headlines(url, source_name, limit):
    feed = feedparser.parse(url)
    return [
        {"title": entry.title, "link": entry.link, "published": entry.get("published", ""), "source": source_name}
        for entry in feed.entries[:limit]
    ]

news_items = fetch_headlines("https://www.cnbc.com/id/10000664/device/rss/rss.html", "CNBC", 8)

data = {
    "indicators": indicators,
    "news": news_items,
    "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
}

with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

print("Successfully updated data.json using official FRED data!")