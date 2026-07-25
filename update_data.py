import os
from dotenv import load_dotenv
load_dotenv()
import json
from datetime import datetime, timezone
from fredapi import Fred
import feedparser

# Initialize FRED with your API key
fred = Fred(api_key=os.environ["FRED_API_KEY"])

# Try to load yesterday's data (if it exists) so we can compare
try:
    with open("data.json", "r") as f:
        previous_data = json.load(f)
except FileNotFoundError:
    previous_data = {}

# Fetch last 30 days of US 10Y yield history for the chart
us_10y_history_raw = fred.get_series("DGS10").dropna().tail(30)
us_10y_history = [
    {"date": date.strftime("%Y-%m-%d"), "value": round(float(value), 2)}
    for date, value in us_10y_history_raw.items()
]
# UK 10Y yield history (monthly data, so 12 = last 12 months)
uk_10y_history_raw = fred.get_series("IRLTLT01GBM156N").dropna().tail(12)
uk_10y_history = [
    {"date": date.strftime("%Y-%m-%d"), "value": round(float(value), 2)}
    for date, value in uk_10y_history_raw.items()
]

# Eurozone 10Y government bond yield history (monthly data, so 12 = last 12 months)
eurozone_10y_history_raw = fred.get_series("IRLTLT01EZM156N").dropna().tail(12)
eurozone_10y_history = [
    {"date": date.strftime("%Y-%m-%d"), "value": round(float(value), 2)}
    for date, value in eurozone_10y_history_raw.items()
]
# Fetch latest market news headlines from CNBC's Markets RSS feed
def fetch_headlines(url, source_name, limit):
    feed = feedparser.parse(url)
    return [
        {
            "title": entry.title,
            "link": entry.link,
            "published": entry.get("published", ""),
            "source": source_name
        }
        for entry in feed.entries[:limit]
    ]

news_items = fetch_headlines("https://www.cnbc.com/id/10000664/device/rss/rss.html", "CNBC", 8)

data = {
    "us_10y": round(float(fred.get_series("DGS10").dropna().iloc[-1]), 2),
    "uk_10y": round(float(fred.get_series("IRLTLT01GBM156N").dropna().iloc[-1]), 2),
    "fed_rate": round(float(fred.get_series("FEDFUNDS").dropna().iloc[-1]), 2),
    "ecb_rate": round(float(fred.get_series("ECBDFR").dropna().iloc[-1]), 2),
    "cpi": round(float(fred.get_series("CPIAUCSL").dropna().iloc[-1]), 2),
    "unemployment": round(float(fred.get_series("UNRATE").dropna().iloc[-1]), 2),
    "uk_cpi": round(float(fred.get_series("GBRCPIALLMINMEI").dropna().iloc[-1]), 2),
    "uk_unemployment": round(float(fred.get_series("LRHUTTTTGBM156S").dropna().iloc[-1]), 2),
    "eurozone_cpi": round(float(fred.get_series("CP0000EZ19M086NEST").dropna().iloc[-1]), 2),
    "eurozone_unemployment": round(float(fred.get_series("LRHUTTTTEZM156S").dropna().iloc[-1]), 2),
    "us_10y_history": us_10y_history,
    "uk_10y_history": uk_10y_history,
    "eurozone_10y_history": eurozone_10y_history,
    "news": news_items,
    "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    }

# Work out trend direction for each field by comparing to the previous value
# Thresholds for what counts as a "notable" move, per indicator
notable_thresholds = {
    "us_10y": 0.05,
    "uk_10y": 0.05,
    "eurozone_10y": 0.05,
    "fed_rate": 0.25,
    "ecb_rate": 0.25,
    "unemployment": 0.2,
    "uk_unemployment": 0.2,
    "eurozone_unemployment": 0.2,
}
percent_based_keys = ["cpi", "uk_cpi", "eurozone_cpi"]

trend = {}
notable = {}
for key in ["us_10y", "uk_10y", "fed_rate", "ecb_rate", "cpi", "unemployment", "uk_cpi", "uk_unemployment", "eurozone_cpi", "eurozone_unemployment"]:
    old_value = previous_data.get(key)
    new_value = data[key]

    if old_value is None:
        trend[key] = "same"
        notable[key] = False
        continue

    if new_value > old_value:
        trend[key] = "up"
    elif new_value < old_value:
        trend[key] = "down"
    else:
        trend[key] = "same"

    change = abs(new_value - old_value)

    if key in percent_based_keys:
        percent_change = (change / old_value) * 100
        notable[key] = percent_change >= 0.3
    else:
        threshold = notable_thresholds.get(key, 999)
        notable[key] = change >= threshold

data["trend"] = trend
data["notable"] = notable

with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

print("Successfully updated data.json using official FRED data!")