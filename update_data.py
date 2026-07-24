import os
from dotenv import load_dotenv
load_dotenv()
import json
from datetime import datetime
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
    "us_10y_history": us_10y_history,
    "news": news_items,
    "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
}

# Work out trend direction for each field by comparing to the previous value
trend = {}
for key in ["us_10y", "uk_10y", "fed_rate", "ecb_rate", "cpi", "unemployment"]:
    old_value = previous_data.get(key)
    new_value = data[key]
    if old_value is None:
        trend[key] = "same"  # no previous data to compare yet
    elif new_value > old_value:
        trend[key] = "up"
    elif new_value < old_value:
        trend[key] = "down"
    else:
        trend[key] = "same"

data["trend"] = trend

with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

print("Successfully updated data.json using official FRED data!")