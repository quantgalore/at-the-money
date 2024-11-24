# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlalchemy
import mysql.connector

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

trading_dates = calendar.schedule(start_date = "2023-05-01", end_date = (datetime.today()+timedelta(days = 1))).index.strftime("%Y-%m-%d").values

trade_time = "09:31"

ticker = "SPY"
index_ticker = "I:SPX"
options_ticker = "SPX"

spread_width = 1

data_list = []
times = []

# date = trading_dates[1:-1][-2]
for date in trading_dates[1:-1]:
    
    try:
        
        start_time = datetime.now()
    
        prior_day = trading_dates[np.where(trading_dates==date)[0][0]-1]    
        next_day = trading_dates[trading_dates > date][0]

        big_underlying_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2015-01-01/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        big_underlying_data.index = pd.to_datetime(big_underlying_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        
        index_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{index_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        index_data.index = pd.to_datetime(index_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        index_data = index_data[(index_data.index.time >= pd.Timestamp(trade_time).time()) & (index_data.index.time <= pd.Timestamp("16:00").time())].copy()
        
        concatenated_regime_dataset = big_underlying_data.copy()
        
        concatenated_regime_dataset["overnight_move_raw"] = (round(((concatenated_regime_dataset["o"] - concatenated_regime_dataset["c"].shift(1)) / concatenated_regime_dataset["c"].shift(1))*100, 2))
        concatenated_regime_dataset["overnight_move_abs"] = abs(concatenated_regime_dataset["overnight_move_raw"])
        concatenated_regime_dataset["avg_move"] = concatenated_regime_dataset["overnight_move_abs"].rolling(window=60, closed = "left").mean()
        concatenated_regime_dataset["overnight_move"] = concatenated_regime_dataset.apply(underlying_return_classification, axis = 1)

        prediction = Model.predict(concatenated_regime_dataset[["overnight_move"]].tail(1))[0]

        direction = prediction
        
        price = index_data["c"].iloc[0]
        
        minute_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = pd.Timestamp(trade_time).time().hour, minutes = (pd.Timestamp(trade_time)).time().minute))
        quote_timestamp = minute_timestamp.value
        close_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = 16, minutes = 0)).value
        
        exp_date = date
        
        if direction == 0:
            
            valid_calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={options_ticker}&contract_type=call&as_of={date}&expiration_date={exp_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
            valid_calls = valid_calls[valid_calls["ticker"].str.contains("SPXW")].copy()
            valid_calls["days_to_exp"] = (pd.to_datetime(valid_calls["expiration_date"]) - pd.to_datetime(date)).dt.days
            valid_calls["distance_from_price"] = abs(valid_calls["strike_price"] - price)
            
            otm_calls = valid_calls[valid_calls["strike_price"] >= (price-5)].copy().sort_values(by="strike_price", ascending=True)
            
            short_call = otm_calls.iloc[[0]]
            long_call = otm_calls.iloc[[spread_width]]
            
            short_strike = short_call["strike_price"].iloc[0]
            short_ticker = short_call["ticker"].iloc[0]
            long_strike = long_call["strike_price"].iloc[0]
            long_ticker  = long_call["ticker"].iloc[0]
            
            real_width = abs(short_strike - long_strike)
            
            short_call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{short_call['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
            short_call_quotes.index = pd.to_datetime(short_call_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
            short_call_quotes["mid_price"] = round((short_call_quotes["bid_price"] + short_call_quotes["ask_price"]) / 2, 2)
            short_call_quotes = short_call_quotes[short_call_quotes.index.strftime("%Y-%m-%d %H:%M") <= minute_timestamp.strftime("%Y-%m-%d %H:%M")].copy()
            
            short_call_quote = short_call_quotes.median(numeric_only=True).to_frame().copy().T
            short_call_quote["t"] = minute_timestamp.strftime("%Y-%m-%d %H:%M")
            
            long_call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{long_call['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
            long_call_quotes.index = pd.to_datetime(long_call_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
            long_call_quotes["mid_price"] = round((long_call_quotes["bid_price"] + long_call_quotes["ask_price"]) / 2, 2)
            long_call_quotes = long_call_quotes[long_call_quotes.index.strftime("%Y-%m-%d %H:%M") <= minute_timestamp.strftime("%Y-%m-%d %H:%M")].copy()
            
            long_call_quote = long_call_quotes.median(numeric_only=True).to_frame().copy().T
            long_call_quote["t"] = minute_timestamp.strftime("%Y-%m-%d %H:%M")
             
            spread = pd.concat([short_call_quote.add_prefix("short_call_"), long_call_quote.add_prefix("long_call_")], axis = 1).dropna()
            
            spread["spread_value"] = spread["short_call_mid_price"] - spread["long_call_mid_price"]
            cost = spread["spread_value"].iloc[0]
            max_loss = abs(short_call["strike_price"].iloc[0] - long_call["strike_price"].iloc[0]) - cost
              
        elif direction == 1:
        
            valid_puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={options_ticker}&contract_type=put&as_of={date}&expiration_date={exp_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
            valid_puts = valid_puts[valid_puts["ticker"].str.contains("SPXW")].copy()
            valid_puts["days_to_exp"] = (pd.to_datetime(valid_puts["expiration_date"]) - pd.to_datetime(date)).dt.days
            valid_puts["distance_from_price"] = abs(price - valid_puts["strike_price"])
            
            otm_puts = valid_puts[valid_puts["strike_price"] <= (price+5)].sort_values("strike_price", ascending = False)
            
            short_put = otm_puts.iloc[[0]]
            long_put = otm_puts.iloc[[spread_width]]
            
            short_strike = short_put["strike_price"].iloc[0]
            short_ticker = short_put["ticker"].iloc[0]
            long_strike = long_put["strike_price"].iloc[0]
            long_ticker  = long_put["ticker"].iloc[0]
            
            real_width = abs(short_strike - long_strike)
        
            short_put_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{short_put['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
            short_put_quotes.index = pd.to_datetime(short_put_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
            short_put_quotes["mid_price"] = round((short_put_quotes["bid_price"] + short_put_quotes["ask_price"]) / 2, 2)
            short_put_quotes = short_put_quotes[short_put_quotes.index.strftime("%Y-%m-%d %H:%M") <= minute_timestamp.strftime("%Y-%m-%d %H:%M")].copy()
            
            short_put_quote = short_put_quotes.median(numeric_only=True).to_frame().copy().T
            short_put_quote["t"] = minute_timestamp.strftime("%Y-%m-%d %H:%M")
            
            long_put_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{long_put['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
            long_put_quotes.index = pd.to_datetime(long_put_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
            long_put_quotes["mid_price"] = round((long_put_quotes["bid_price"] + long_put_quotes["ask_price"]) / 2, 2)
            long_put_quotes = long_put_quotes[long_put_quotes.index.strftime("%Y-%m-%d %H:%M") <= minute_timestamp.strftime("%Y-%m-%d %H:%M")].copy()
            
            long_put_quote = long_put_quotes.median(numeric_only=True).to_frame().copy().T
            long_put_quote["t"] = minute_timestamp.strftime("%Y-%m-%d %H:%M")

            spread = pd.concat([short_put_quote.add_prefix("short_put_"), long_put_quote.add_prefix("long_put_")], axis = 1).dropna()
            
            spread["spread_value"] = spread["short_put_mid_price"] - spread["long_put_mid_price"]
            cost = spread["spread_value"].iloc[0]
            max_loss = abs(short_put["strike_price"].iloc[0] - long_put["strike_price"].iloc[0]) - cost
      
        closing_value = index_data["c"].iloc[-1]
        
        open_price = cost
        
        if direction == 1:
            settlement = closing_value - short_strike
            if settlement > 0:
                settlement = 0
                final_pnl = open_price
            else:
                final_pnl = settlement + open_price
                
        elif direction == 0:
            settlement = short_strike - closing_value
            if settlement > 0:
                settlement = 0
                final_pnl = open_price
            else:
                final_pnl = settlement + open_price
                
        gross_pnl = np.maximum(final_pnl, max_loss*-1)
        
        vol_data = pd.DataFrame([{"date": date, "direction": direction, "actual_dir": np.int64(closing_value > price),
                                  "credit": open_price, "gross_pnl": gross_pnl}])
        
        data_list.append(vol_data)
        
        end_time = datetime.now()
        
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((np.where(trading_dates==date)[0][0]/len(trading_dates))*100,2)
        iterations_remaining = len(trading_dates) - np.where(trading_dates==date)[0][0]
        average_time_to_complete = np.mean(times)
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
            
    except Exception as error:
        print(error, date)
        continue

full_data = pd.concat(data_list).drop_duplicates(subset = "date", keep = "last").sort_values(by="date", ascending=True)

wins = full_data[full_data["gross_pnl"] > 0].copy()
losses = full_data[full_data["gross_pnl"] < 0].copy()

avg_win = wins["gross_pnl"].mean()
avg_loss = losses["gross_pnl"].mean()

win_rate = len(full_data[full_data["gross_pnl"] > 0]) / len(full_data)

expected_value = (avg_win * win_rate) + ((1-win_rate) * avg_loss)

full_data["capital"] = 8000 + ((full_data["gross_pnl"] - 0.04)*100).cumsum()

plt.figure(dpi=200)
plt.xticks(rotation=45)
plt.title(f"Selling 0-DTE Credit Spreads - Synthetic Direction Data")
plt.plot(pd.to_datetime(full_data["date"]), full_data["capital"])
plt.legend(["Net PnL Including Fees"])
plt.show()
plt.close()