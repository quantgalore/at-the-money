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
    
def binarizer(x):
    if x > 0:
        return 1
    else:
        return 0 

######

polygon_api_key = "polygon.io api key, use 'QUANTGALORE' for 10% off."
calendar = get_calendar("NYSE")

# 2023-05-01 is about the start of SPX coverage via polygon. 
trading_dates = calendar.schedule(start_date = "2023-05-01", end_date = (datetime.today()-timedelta(days = 1))).index.strftime("%Y-%m-%d").values

data_list = []

trade_time = "09:30"
close_time = "16:00"
minutes_in_between = np.array(pd.date_range(start=trade_time, end="15:59", freq="T").strftime("%H:%M"))
minutes_thereafter = minutes_in_between[minutes_in_between > trade_time]

ticker = "SPY"
index_ticker = "I:SPX"
options_ticker = "SPX"

spread_width = 1

# If the probability of the spread gets to this amount, then hedge. 0.25 = 25%
adjustment_implied_prob = .25

# 1 = sell enough spreads to reduce 100% of the original max risk. 0.25 = 25%, 1.5 = get 50% more, and so on. 
risk_reduction = 1

# date = trading_dates[1:-1][0]
for date in trading_dates[1:-1]:
    
    try:
    
        prior_day = trading_dates[np.where(trading_dates==date)[0][0]-1]    
        next_day = trading_dates[trading_dates > date][0]

        big_underlying_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2015-01-01/{prior_day}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        big_underlying_data.index = pd.to_datetime(big_underlying_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        
        index_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{index_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        index_data.index = pd.to_datetime(index_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        index_data = index_data[(index_data.index.time >= pd.Timestamp(trade_time).time()) & (index_data.index.time <= pd.Timestamp("16:00").time())].copy()

        price = index_data["c"].iloc[0]
        
        minute_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = pd.Timestamp(trade_time).time().hour, minutes = (pd.Timestamp(trade_time) + timedelta(minutes=1)).time().minute))
        quote_timestamp = minute_timestamp.value
        close_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = 16, minutes = 0)).value
        
        exp_date = date
                
        # =============================================================================
        # Start off with a put spread
        # =============================================================================
        
        valid_puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={options_ticker}&contract_type=put&as_of={date}&expiration_date={exp_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
        valid_puts = valid_puts[valid_puts["ticker"].str.contains("SPXW")].copy()
        valid_puts["days_to_exp"] = (pd.to_datetime(valid_puts["expiration_date"]) - pd.to_datetime(date)).dt.days
        valid_puts["distance_from_price"] = abs(price - valid_puts["strike_price"])
        
        otm_puts = valid_puts[valid_puts["strike_price"] <= (price+5)].sort_values("strike_price", ascending = False)
        
        short_put = otm_puts.iloc[[0]]
        long_put = otm_puts.iloc[[spread_width]]
        
        short_put_strike = short_put["strike_price"].iloc[0]
        short_put_ticker = short_put["ticker"].iloc[0]
        long_put_strike = long_put["strike_price"].iloc[0]
        long_put_ticker  = long_put["ticker"].iloc[0]
        
        real_width = abs(short_put_strike - long_put_strike)
    
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
        put_cost = spread["spread_value"].iloc[0]
        put_max_loss = abs(short_put["strike_price"].iloc[0] - long_put["strike_price"].iloc[0]) - put_cost
            
        # =============================================================================
        # Pull forward prices such that you know if/when to sell opposite spread  
        # =============================================================================
        
        quotes_list = []
        
        post_trade_minute = minutes_thereafter[0]
        
        post_trade_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = pd.Timestamp(post_trade_minute).time().hour, minutes = pd.Timestamp(post_trade_minute).time().minute))
        post_trade_quote_timestamp = post_trade_timestamp.value
        
        close_trade_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = pd.Timestamp(close_time).time().hour, minutes = pd.Timestamp(close_time).time().minute)).value
    
        ## Short Put
        
        post_short_put_data_list = []
        post_short_put_next_urls = []
        
        post_short_put_quote_request_0 = requests.get(f"https://api.polygon.io/v3/quotes/{short_put['ticker'].iloc[0]}?timestamp.gte={post_trade_quote_timestamp}&timestamp.lt={close_trade_timestamp}&order=asc&limit=50000&sort=timestamp&apiKey={polygon_api_key}").json()   
        post_short_put_data = pd.json_normalize(post_short_put_quote_request_0["results"]).set_index("sip_timestamp")
        post_short_put_data.index = pd.to_datetime(post_short_put_data.index, unit="ns", utc=True).tz_convert("America/New_York")
        post_short_put_data["mid_price"] = round((post_short_put_data["bid_price"] + post_short_put_data["ask_price"]) / 2, 2)
        
        post_short_put_data_list.append(post_short_put_data)
        post_short_put_next_urls.append(post_short_put_quote_request_0["next_url"])
        
        # request_iteration = 1
        for request_iteration in range(1, 100):
            
            try:
                post_short_put_quote_request_n = requests.get(f"{post_short_put_next_urls[-1]}&order=asc&limit=50000&sort=timestamp&apiKey={polygon_api_key}").json()
                post_short_put_data = pd.json_normalize(post_short_put_quote_request_n["results"]).set_index("sip_timestamp")
                post_short_put_data.index = pd.to_datetime(post_short_put_data.index, unit="ns", utc=True).tz_convert("America/New_York")
                post_short_put_data["mid_price"] = round((post_short_put_data["bid_price"] + post_short_put_data["ask_price"]) / 2, 2)
                
                post_short_put_data_list.append(post_short_put_data)
                post_short_put_next_urls.append(post_short_put_quote_request_n["next_url"])
                
            except (Exception, KeyError):
                break
        
        full_short_put_data = pd.concat(post_short_put_data_list)
        full_short_put_data["minute_timestamp"] = full_short_put_data.index.strftime("%Y-%m-%d %H:%M")
        
        grouped_short_put_data = full_short_put_data.groupby("minute_timestamp").median().copy()
        
        ## Long Put
        
        post_long_put_data_list = []
        post_long_put_next_urls = []

        post_long_put_quote_request_0 = requests.get(f"https://api.polygon.io/v3/quotes/{long_put['ticker'].iloc[0]}?timestamp.gte={post_trade_quote_timestamp}&timestamp.lt={close_trade_timestamp}&order=asc&limit=50000&sort=timestamp&apiKey={polygon_api_key}").json()   
        post_long_put_data = pd.json_normalize(post_long_put_quote_request_0["results"]).set_index("sip_timestamp")
        post_long_put_data.index = pd.to_datetime(post_long_put_data.index, unit="ns", utc=True).tz_convert("America/New_York")
        post_long_put_data["mid_price"] = round((post_long_put_data["bid_price"] + post_long_put_data["ask_price"]) / 2, 2)

        post_long_put_data_list.append(post_long_put_data)
        post_long_put_next_urls.append(post_long_put_quote_request_0["next_url"])

        # request_iteration = 1
        for request_iteration in range(1, 100):
            
            try:
                post_long_put_quote_request_n = requests.get(f"{post_long_put_next_urls[-1]}&order=asc&limit=50000&sort=timestamp&apiKey={polygon_api_key}").json()
                post_long_put_data = pd.json_normalize(post_long_put_quote_request_n["results"]).set_index("sip_timestamp")
                post_long_put_data.index = pd.to_datetime(post_long_put_data.index, unit="ns", utc=True).tz_convert("America/New_York")
                post_long_put_data["mid_price"] = round((post_long_put_data["bid_price"] + post_long_put_data["ask_price"]) / 2, 2)
                
                post_long_put_data_list.append(post_long_put_data)
                post_long_put_next_urls.append(post_long_put_quote_request_n["next_url"])
                
            except (Exception, KeyError):
                break

        full_long_put_data = pd.concat(post_long_put_data_list)
        full_long_put_data["minute_timestamp"] = full_long_put_data.index.strftime("%Y-%m-%d %H:%M")

        grouped_long_put_data = full_long_put_data.groupby("minute_timestamp").median().copy()
        
        ###
        
        grouped_spread_data = pd.concat([grouped_short_put_data.add_prefix("short_put_"), grouped_long_put_data.add_prefix("long_put_")], axis = 1).dropna()
        grouped_spread_data["spread_value"] = grouped_spread_data["short_put_mid_price"] - grouped_spread_data["long_put_mid_price"]
        grouped_spread_data["spread_nat_value"] = grouped_spread_data["short_put_bid_price"] - grouped_spread_data["long_put_ask_price"]
        
        grouped_spread_data["implied_prob"] = round((real_width - grouped_spread_data["spread_value"]) / real_width, 2)
        
        hedge_trigger = grouped_spread_data[grouped_spread_data["implied_prob"] <= adjustment_implied_prob].copy()
        
        if len(hedge_trigger) < 1:
            
            closing_value = index_data["c"].iloc[-1]
                   
            settlement = closing_value - short_put_strike
            if settlement > 0:
                settlement = 0
                final_pnl = put_cost
            else:
                final_pnl = np.maximum(settlement + put_cost, put_max_loss*-1)
                
            trade_data = pd.DataFrame([{"date": date, "base_credit": put_cost, "hedge_credit": 0, 
                                         "base_pnl": final_pnl, "hedge_pnl": 0,
                                         "gross_pnl": final_pnl}])
            
            data_list.append(trade_data)
            
            continue
        
        hedge_minute_time = pd.to_datetime(hedge_trigger.index[0]).tz_localize("America/New_York")
        hedge_timestamp = pd.to_datetime(hedge_trigger.index[0]).tz_localize("America/New_York").value
        
        valid_calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={options_ticker}&contract_type=call&as_of={date}&expiration_date={exp_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
        valid_calls = valid_calls[valid_calls["ticker"].str.contains("SPXW")].copy()
        # valid_calls["days_to_exp"] = (pd.to_datetime(valid_calls["expiration_date"]) - pd.to_datetime(date)).dt.days
        # valid_calls["distance_from_price"] = abs(valid_calls["strike_price"] - price)
        
        otm_calls = valid_calls[valid_calls["strike_price"] >= short_put_strike].copy().sort_values(by="strike_price", ascending=True)
        
        short_call = otm_calls.iloc[[0]]
        long_call = otm_calls.iloc[[spread_width]]
        
        short_call_strike = short_call["strike_price"].iloc[0]
        short_call_ticker = short_call["ticker"].iloc[0]
        long_call_strike = long_call["strike_price"].iloc[0]
        long_call_ticker  = long_call["ticker"].iloc[0]
        
        real_width = abs(short_call_strike - long_call_strike)
        
        short_call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{short_call['ticker'].iloc[0]}?timestamp.gte={hedge_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
        short_call_quotes.index = pd.to_datetime(short_call_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
        short_call_quotes["mid_price"] = round((short_call_quotes["bid_price"] + short_call_quotes["ask_price"]) / 2, 2)
        short_call_quotes = short_call_quotes[short_call_quotes.index.strftime("%Y-%m-%d %H:%M") <= hedge_minute_time.strftime("%Y-%m-%d %H:%M")].copy()
        
        short_call_quote = short_call_quotes.median(numeric_only=True).to_frame().copy().T
        short_call_quote["t"] = hedge_minute_time.strftime("%Y-%m-%d %H:%M")
        
        long_call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{long_call['ticker'].iloc[0]}?timestamp.gte={hedge_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
        long_call_quotes.index = pd.to_datetime(long_call_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
        long_call_quotes["mid_price"] = round((long_call_quotes["bid_price"] + long_call_quotes["ask_price"]) / 2, 2)
        long_call_quotes = long_call_quotes[long_call_quotes.index.strftime("%Y-%m-%d %H:%M") <= hedge_minute_time.strftime("%Y-%m-%d %H:%M")].copy()
        
        long_call_quote = long_call_quotes.median(numeric_only=True).to_frame().copy().T
        long_call_quote["t"] = hedge_minute_time.strftime("%Y-%m-%d %H:%M")
         
        spread = pd.concat([short_call_quote.add_prefix("short_call_"), long_call_quote.add_prefix("long_call_")], axis = 1).dropna()
        
        spread["spread_value"] = spread["short_call_mid_price"] - spread["long_call_mid_price"]
        hedge_cost = spread["spread_value"].iloc[0]
        hedge_max_loss = abs(short_call["strike_price"].iloc[0] - long_call["strike_price"].iloc[0]) - hedge_cost
        
        # =============================================================================
        # PnL Calcs.
        # =============================================================================
        
        contracts_to_fully_hedge = np.ceil(put_max_loss*risk_reduction / hedge_cost)
        
        closing_value = index_data["c"].iloc[-1]
               
        settlement = closing_value - short_put_strike
        if settlement > 0:
            settlement = 0
            final_pnl = put_cost
        else:
            final_pnl = np.maximum(settlement + put_cost, put_max_loss*-1)
        
        hedge_settlement =  short_call_strike - closing_value
        if hedge_settlement > 0:
            hedge_settlement = 0
            hedge_final_pnl = hedge_cost
        else:
            hedge_final_pnl = np.maximum(hedge_settlement + hedge_cost, hedge_max_loss*-1) 
            
        combined_pnl = final_pnl + (hedge_final_pnl * contracts_to_fully_hedge)
       
        trade_data = pd.DataFrame([{"date": date, "base_credit": put_cost, "hedge_credit": hedge_cost, 
                                     "base_pnl": final_pnl, "hedge_pnl": hedge_final_pnl*contracts_to_fully_hedge,
                                     "gross_pnl": combined_pnl}])
        
        data_list.append(trade_data)
        
    except Exception as error:
        print(error)
        continue
    
full_data = pd.concat(data_list)

wins = full_data[full_data["gross_pnl"] > 0].copy()
losses = full_data[full_data["gross_pnl"] < 0].copy()

avg_win = wins["gross_pnl"].mean()
avg_loss = losses["gross_pnl"].mean()

win_rate = len(full_data[full_data["gross_pnl"] > 0]) / len(full_data)

expected_value = (avg_win * win_rate) + ((1-win_rate) * avg_loss)

full_data["capital"] = 8000 + ((full_data["gross_pnl"] - 0.08)*100).cumsum()
full_data["base_capital"] = 8000 + ((full_data["base_pnl"] - 0.04)*100).cumsum()

plt.figure(dpi=200)
plt.xticks(rotation=45)
plt.title("Performance of Hedging a 0-DTE ATM Put Credit Spread")
plt.plot(pd.to_datetime(full_data["date"]), full_data["capital"])
plt.plot(pd.to_datetime(full_data["date"]), full_data["base_capital"])
plt.legend(["hedging - net pnl", "no hedging - net pnl"])
plt.show()
plt.close()