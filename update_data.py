import os
from dotenv import load_dotenv
load_dotenv()
import json
from datetime import datetime
from fredapi import Fred

# Initialize FRED with your API key
fred = Fred(api_key=os.environ["FRED_API_KEY"])

# Try to load yesterday's data (if it exists) so we can compare
try:
    with open("data.json", "r") as f:
        previous_data = json.load(f)
except FileNotFoundError:
    previous_data = {}

data = {
    "us_10y": round(float(fred.get_series("DGS10").dropna().iloc[-1]), 2),
    "uk_10y": round(float(fred.get_series("IRLTLT01GBM156N").dropna().iloc[-1]), 2),
    "fed_rate": round(float(fred.get_series("FEDFUNDS").dropna().iloc[-1]), 2),
    "ecb_rate": round(float(fred.get_series("ECBDFR").dropna().iloc[-1]), 2),
    "cpi": round(float(fred.get_series("CPIAUCSL").dropna().iloc[-1]), 2),
    "unemployment": round(float(fred.get_series("UNRATE").dropna().iloc[-1]), 2),
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