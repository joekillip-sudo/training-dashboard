import os
from dotenv import load_dotenv
load_dotenv()
import json
from datetime import datetime
from fredapi import Fred

# Initialize FRED with your API key
fred = Fred(api_key=os.environ["FRED_API_KEY"])

# Fetch official macro data using verified FRED Series IDs
data = {
    "us_10y": round(float(fred.get_series("DGS10").dropna().iloc[-1]), 2),
    "uk_10y": round(float(fred.get_series("IRLTLT01GBM156N").dropna().iloc[-1]), 2),
    "fed_rate": round(float(fred.get_series("FEDFUNDS").dropna().iloc[-1]), 2),
    "ecb_rate": round(float(fred.get_series("ECBDFR").dropna().iloc[-1]), 2),
    "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
}

# Save to data.json
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

print("Successfully updated data.json using official FRED data!")