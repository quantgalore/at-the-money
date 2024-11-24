# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

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

features = ["overnight_move"]
target = "dir"

X = synth_dataset[features]
Y = synth_dataset[target]

Model = RandomForestClassifier().fit(X, Y)

######

polygon_api_key = "polygon.io api key, use 'QUANTGALORE' for 10% off."
calendar = get_calendar("NYSE")

trading_dates = calendar.schedule(start_date = "2019-01-01", end_date = (datetime.today())).index.strftime("%Y-%m-%d").values

ticker = "SPY"

giant_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2024-01-01/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
giant_data.index = pd.to_datetime(giant_data.index, unit="ms", utc=True).tz_convert("America/New_York")
giant_data["date"] = giant_data.index.strftime("%Y-%m-%d")

giant_data["overnight_move_raw"] = (round(((giant_data["o"] - giant_data["c"].shift(1)) / giant_data["c"].shift(1))*100, 2))
giant_data["overnight_move_abs"] = abs(giant_data["overnight_move_raw"])
giant_data["avg_move"] = giant_data["overnight_move_abs"].rolling(window=60, closed = "left").mean()

giant_data["overnight_move"] = giant_data.apply(underlying_return_classification, axis = 1)

giant_data["intraday_move"] = (round(((giant_data["c"] - giant_data["o"]) / giant_data["o"])*100, 2))
giant_data["intraday_move"] = giant_data["intraday_move"].apply(binarizer)

giant_data = giant_data.dropna().copy()

giant_data["pred"] = Model.predict(giant_data[["overnight_move"]])
giant_data["accurate"] = np.int64(giant_data["pred"] == giant_data["intraday_move"])
giant_data["rolling_win_rate"] = giant_data["accurate"].rolling(window=10, closed = "left").apply(lambda x: len(x[x == 1]) / 10)

daily_prediction = giant_data[giant_data["date"] == trading_dates[-1]]["pred"].iloc[0]
rolling_win_rate = giant_data[giant_data["date"] == trading_dates[-1]]["rolling_win_rate"].iloc[0]

if rolling_win_rate < .6:
    print(f"No trade today, win rate too low ({round(rolling_win_rate, 2)})")
    # sys.exit()
    
else:
    
    if daily_prediction == 0:
        print(f"Sell Calls, {trading_dates[-1]}")
    elif daily_prediction == 1:
        print(f"Sell Puts, {trading_dates[-1]}")
