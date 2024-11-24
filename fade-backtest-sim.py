# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from sklearn.ensemble import RandomForestClassifier

def underlying_return_classification(row):
    if row["overnight_move_raw"] > row["avg_move"]:
        return 2
    elif (row["overnight_move_raw"] > 0) and (row["overnight_move_raw"] <= row["avg_move"]):
        return 1
    elif (row["overnight_move_raw"] < 0) and (row["overnight_move_raw"] > -row["avg_move"]):
        return -1
    elif row["overnight_move_raw"] < -row["avg_move"]:
        return -2
    else:
        return 0  # Optional default case if none of the conditions match
    
def binarizer(x):
    if x > 0:
        return 1
    else:
        return 0 

######

synth_dataset = pd.read_csv("synth_fade_data.csv")

# features = ["overnight_move", "regime", "vol_returns", "safe_haven_returns"]
features = ["overnight_move"]
target = "dir"

X = synth_dataset[features]
Y = synth_dataset[target]

# Model = KNeighborsClassifier(n_neighbors=1).fit(X, Y)
Model = RandomForestClassifier().fit(X, Y)

######

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

trading_dates = calendar.schedule(start_date = "2000-01-01", end_date = (datetime.today()-timedelta(days = 1))).index.strftime("%Y-%m-%d").values

liquid_tickers_original = pd.read_csv("liquid_tickers.csv")
liquid_tickers = liquid_tickers_original["ticker"].values

ticker = "SPY"

window_period = 60

giant_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2000-01-01/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
giant_data.index = pd.to_datetime(giant_data.index, unit="ms", utc=True).tz_convert("America/New_York")
giant_data["date"] = giant_data.index.strftime("%Y-%m-%d")

giant_data["overnight_move_raw"] = (round(((giant_data["o"] - giant_data["c"].shift(1)) / giant_data["c"].shift(1))*100, 2))
giant_data["overnight_move_abs"] = abs(giant_data["overnight_move_raw"])

giant_data["avg_move"] = giant_data["overnight_move_abs"].rolling(window=window_period, closed = "left").mean()
giant_data["overnight_move"] = giant_data.apply(underlying_return_classification, axis = 1)

giant_data["intraday_move"] = (round(((giant_data["c"] - giant_data["o"]) / giant_data["o"])*100, 2))
giant_data["intraday_move"] = giant_data["intraday_move"].apply(binarizer)

giant_data = giant_data.dropna().copy()

giant_data["pred"] = Model.predict(giant_data[["overnight_move"]])
giant_data["proba"] = Model.predict_proba(giant_data[["overnight_move"]])[0][giant_data["pred"]]

giant_data["gross_pnl"] = giant_data.apply(lambda x: 2.5 if x["pred"] == x["intraday_move"] else -2.5, axis = 1)

full_data = giant_data.copy()

# un-comment to toggle the use of rolling win rates. 

# full_data["rolling_win_rate"] = full_data["gross_pnl"].rolling(window=10, closed="left").apply(lambda x: len(x[x>0]) / 10)
# full_data = full_data[full_data["rolling_win_rate"] >= .6].copy()

full_data["capital"] = 8000 + ((full_data["gross_pnl"] - 0.04)*100).cumsum()


plt.figure(dpi=200)
plt.xticks(rotation=45)
plt.title(f"Simulated 0-DTE Spreads - Synthetic")
plt.suptitle(f"Only when model is 'hot' - won > 6 of last 10")
plt.plot(full_data.index, full_data["capital"])
plt.show()
plt.close()