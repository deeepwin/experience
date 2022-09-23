---
layout: post
title:  " Study comparing Yahoo Intraday with Daily Closing Prices"
date:   2022-09-22 17:49:22 +0100
categories: data
comments_id: 2
---
Time to read this post: 10 mins

## Background

There is not much information on the internet in how Yahoo daily closing prices are calculated and how they compare to the intraday data. At the NYSE daily closing price is determine by an [closing auction](https://www.nyse.com/article/nyse-closing-auction-insiders-guide). In this article we make an attempt to compare real data. The goal ist to compare:

1) The day end prices of historical intraday data such as 1m, 5m and 30m data to the daily closing prices
2) Compare live intraday data with the historical intraday data
 

## Imports and Settings


```python
import warnings
warnings.filterwarnings('ignore')
```


```python

import os
import pandas as pd
import numpy as np

import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime
from datetime import timedelta

import yfinance as yf

pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

%load_ext autoreload
%autoreload 2
%cd ..

data_from_date  = datetime.datetime.now() - pd.offsets.Day(29) # Yahoo allows to download last 30 days only
data_end_date   = datetime.datetime.now() - pd.offsets.BDay(2)
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    /home/martin/GitHub



```python

from alpaca_trade_api.rest import REST, TimeFrame

os.environ['APCA_API_BASE_URL']     = 'https://paper-api.alpaca.markets'
os.environ['APCA_API_KEY_ID']       = ''
os.environ['APCA_API_SECRET_KEY']   = ''

alpaca_api = REST()
account = alpaca_api.get_account()
```

## Compare historical Daily Closing Prices with Intraday Prices

Define the tickers you want to compare.


```python

tickers = ['CL', 'CSCO', 'FAST', 'HOLX', 'HSY', 'INTU', 'NKE', 'NTAP', 'TER', 'TSCO', 'TXN', 'WAT', 'YUM']
```

Define metric to use for comparison.


```python
metric = 'adj_close'
```

### Download Prices


```python
df_dates = yf.download([tickers[0]], interval='1D', start=data_from_date, end=data_end_date, progress=False)
dates = df_dates.index
```


```python
df_data = pd.DataFrame(index=dates, columns=['d_y', '1m_y', '5m_y', '30m_y', 'vwap_y', '1m_a'], dtype=np.float32)
df_data.index.names = ['date']

df_diffs = pd.DataFrame(columns=['%_1m_y', '%_5m_y', '%_30m_y', '%_vwap_y', '%_1m_a'], dtype=np.float32)

for ticker in tickers:
    
    for date in dates:

        start_date = (date.date() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
        end_date   = (date.date() + pd.offsets.BDay(1)).strftime('%Y-%m-%d')

        df_y_daily = yf.download([ticker], interval='1D', start=date, end=end_date, progress=False)
        df_y_daily.columns = map(str.lower, df_y_daily.columns)
        df_y_daily.rename(columns={'adj close': 'adj_close'}, inplace=True)
        df_y_daily.index.names = ['date']
        df_y_daily = df_y_daily.filter(regex='^' + date.date().strftime('%Y-%m-%d'), axis=0)
        df_y_daily.sort_index(inplace=True)
        df_y_daily.sort_index(inplace=True, axis=1)
        
        df_y_1m = yf.download([ticker], interval='1m', start=date, end=end_date, progress=False)
        df_y_1m.columns = map(str.lower, df_y_1m.columns)
        df_y_1m.rename(columns={'adj close': 'adj_close'}, inplace=True)
        df_y_1m.index.names = ['date']
        df_y_1m = df_y_1m.filter(regex='^' + date.date().strftime('%Y-%m-%d'), axis=0)
        df_y_1m.sort_index(inplace=True)
        df_y_1m.sort_index(inplace=True, axis=1)

        df_y_5m = yf.download([ticker], interval='5m', start=date, end=end_date, progress=False)
        df_y_5m.columns = map(str.lower, df_y_5m.columns)
        df_y_5m.rename(columns={'adj close': 'adj_close'}, inplace=True)
        df_y_5m.index.names = ['date']
        df_y_5m = df_y_5m.filter(regex='^' + date.date().strftime('%Y-%m-%d'), axis=0)
        df_y_5m.sort_index(inplace=True)
        df_y_5m.sort_index(inplace=True, axis=1)

        df_y_30m = yf.download([ticker], interval='30m', start=date, end=end_date, progress=False)
        df_y_30m.columns = map(str.lower, df_y_30m.columns)
        df_y_30m.rename(columns={'adj close': 'adj_close'}, inplace=True)
        df_y_30m.index.names = ['date']
        df_y_30m = df_y_30m.filter(regex='^' + date.date().strftime('%Y-%m-%d'), axis=0)
        df_y_30m.sort_index(inplace=True)
        df_y_30m.sort_index(inplace=True, axis=1)

        df_a_1m_not_adj = alpaca_api.get_bars(ticker, '1min', start_date, end_date, adjustment='raw').df
        df_a_1m_not_adj.drop(df_a_1m_not_adj.columns.difference(['close']), 1, inplace=True)
        df_a_1m = alpaca_api.get_bars(ticker, '1min', start_date, end_date, adjustment='all').df
        df_a_1m.rename(columns={'close': 'adj_close'}, inplace=True)
        df_a_1m = df_a_1m.merge(df_a_1m_not_adj, how='left', left_index=True, right_index=True).sort_index()
        df_a_1m = df_a_1m.filter(regex='^' + date.date().strftime('%Y-%m-%d'), axis=0)
        df_a_1m.sort_index(inplace=True)
        df_a_1m.sort_index(inplace=True, axis=1)
        calendar = alpaca_api.get_calendar(start=start_date, end=end_date)[0]
        df_a_1m = df_a_1m.tz_convert(tz='America/New_York').between_time(calendar.open, datetime.time(15, 59, 0))

        # compute vwap over last 30 minutes with 1m bars
        q       = df_y_1m.volume[-30:]
        p       = df_y_1m[metric][-30:]
        vwap    = (p * q).cumsum() / q.cumsum()

        df_data.loc[date] = [df_y_daily.iloc[-1].adj_close, df_y_1m.iloc[-1][metric], df_y_5m.iloc[-1][metric], df_y_30m.iloc[-1][metric], vwap[-1], df_a_1m.iloc[-1].adj_close]

    # compute percentage change
    df_diff = pd.DataFrame(index=dates, columns=df_diffs.columns, dtype=np.float32)
    nb_of_columns = len(df_diff.columns)
    
    multi_1 = np.broadcast_to(((np.ones(len(dates), dtype=np.float32)*100) / df_data.values[:,0].squeeze()).T, (nb_of_columns, len(dates))).T
    multi_2 = np.broadcast_to(df_data.values[:,0].squeeze(), (nb_of_columns, len(dates))).T - df_data.values[:,1:]

    df_diff.iloc[:] = np.round(multi_1 * multi_2, decimals=3)

    df_diffs = pd.concat([df_diffs, df_diff])
```

### Analyse Result

The table below shows the differences between daily closing price and intraday data in percentage from adjusted closing price.


```python
df_diffs.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_1m_y</th>
      <th>%_5m_y</th>
      <th>%_30m_y</th>
      <th>%_vwap_y</th>
      <th>%_1m_a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>234.000</td>
      <td>234.000</td>
      <td>234.000</td>
      <td>234.000</td>
      <td>234.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.016</td>
      <td>-0.016</td>
      <td>-0.016</td>
      <td>0.029</td>
      <td>-0.004</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.063</td>
      <td>0.063</td>
      <td>0.063</td>
      <td>0.185</td>
      <td>0.022</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.525</td>
      <td>-0.525</td>
      <td>-0.525</td>
      <td>-0.653</td>
      <td>-0.080</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.020</td>
      <td>-0.020</td>
      <td>-0.020</td>
      <td>-0.088</td>
      <td>-0.015</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.003</td>
      <td>-0.003</td>
      <td>-0.003</td>
      <td>0.020</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.158</td>
      <td>0.004</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.088</td>
      <td>0.088</td>
      <td>0.088</td>
      <td>0.470</td>
      <td>0.100</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusions

We can make the following conclusions:

- Worst price difference is -0.73% to daily intraday over last 30 trading days
- Alpaca standard deviation is better with 0.036 compared to Yahoo with 0.13
- VWAP with 1m bars between 15:30 and 16:00 is worse
- Overall, price difference appear not to large.


## Compare Live and Historical Intraday Data

In a first step, we download live data at around 15:45 NY Time on a specific date in this case 2022-09-21.


```python
# define the date when data was downloaded
test_date = pd.Timestamp.today()
```


```python
arrays = [[], []]
tuples = list(zip(*arrays))

multi_index = pd.MultiIndex.from_tuples(tuples, names=["date", "ticker"])
df_y_1m_live = pd.DataFrame(index=multi_index, columns=['open', 'high', 'low', 'adj_close', 'close', 'volume'], dtype=np.float32)
```


```python
start_date  = test_date.date()
end_date    = (test_date.date() + pd.offsets.BDay(1)).strftime('%Y-%m-%d')

for ticker in tickers:

    df_rows = yf.download([ticker], interval='1m', start=start_date, end=end_date, progress=False)
    df_rows.columns = map(str.lower, df_rows.columns)
    df_rows.rename(columns={'adj close': 'adj_close'}, inplace=True)
    df_rows.index.names = ['date']
    df_rows.sort_index(inplace=True)
    df_rows.sort_index(inplace=True, axis=1)

    for time, prices in df_rows.iterrows():

        idx = pd.IndexSlice
        df_y_1m_live.loc[idx[time, ticker], 'open']      = prices.open
        df_y_1m_live.loc[idx[time, ticker], 'high']      = prices.high
        df_y_1m_live.loc[idx[time, ticker], 'low']       = prices.low
        df_y_1m_live.loc[idx[time, ticker], 'adj_close'] = prices.adj_close
        df_y_1m_live.loc[idx[time, ticker], 'close']     = prices.close
        df_y_1m_live.loc[idx[time, ticker], 'volume']    = prices.volume
            
```

Data is stored to Jupyter Notebook. Afterwards we have to wait for one day to continue to download same day as historical dataset.


```python
%store df_y_1m_live test_date
```

On 2022-09-22, restore previous data.


```python
# load previously stored data
%store -r
```


```python
# define how many lines to compare
nb_of_lines = 10
```


```python
from IPython.core.display import HTML

for ticker in tickers:

    df_live_ticker = df_y_1m_live.filter(regex=ticker, axis=0).reset_index().drop('ticker', axis=1).set_index('date').sort_index(axis=1)

    # download historical data for same date
    start_date = (test_date.date() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    end_date   = (test_date.date() + pd.offsets.BDay(1)).strftime('%Y-%m-%d')

    df_y_1m = yf.download([ticker], interval='1m', start=start_date, end=end_date, progress=False)
    df_y_1m.columns = map(str.lower, df_y_1m.columns)
    df_y_1m.rename(columns={'adj close': 'adj_close'}, inplace=True)
    df_y_1m.index.names = ['date']
    df_y_1m = df_y_1m.filter(regex='^' + test_date.date().strftime('%Y-%m-%d'), axis=0)
    df_y_1m.sort_index(inplace=True)
    df_y_1m.sort_index(inplace=True, axis=1)
    
    # combine live and historical data where time of bars match
    merged = df_live_ticker.merge(df_y_1m, how='left', left_index=True, right_index=True).sort_index().dropna()

    print(ticker)

    # compute percentage change
    multi_1 = merged.iloc[:, 0:6].values
    multi_2 = merged.iloc[:, 6:12].values

    diff_merged = pd.DataFrame( 100 / multi_1 * (multi_1-multi_2), 
                                columns=['%_' + elem.replace('_x', '') for elem in merged.columns[0:6]],
                                index=merged.index).dropna()

    diff_merged = diff_merged.loc[~(diff_merged==0).all(axis=1)]
    display(HTML(diff_merged.describe().to_html()))
    
```

    ADBE



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.007</td>
      <td>-0.007</td>
      <td>-0.000</td>
      <td>0.009</td>
      <td>0.012</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.023</td>
      <td>0.023</td>
      <td>0.001</td>
      <td>0.031</td>
      <td>0.041</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.085</td>
      <td>-0.085</td>
      <td>-0.003</td>
      <td>0.000</td>
      <td>-0.032</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-4.334</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.024</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.969</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.112</td>
      <td>0.129</td>
      <td>12.038</td>
    </tr>
  </tbody>
</table>


    ADSK



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.006</td>
      <td>-0.006</td>
      <td>-0.034</td>
      <td>0.019</td>
      <td>-0.021</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.054</td>
      <td>0.054</td>
      <td>0.064</td>
      <td>0.029</td>
      <td>0.090</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.161</td>
      <td>-0.161</td>
      <td>-0.221</td>
      <td>0.000</td>
      <td>-0.311</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.040</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-3.942</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.273</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.005</td>
      <td>0.005</td>
      <td>0.000</td>
      <td>0.048</td>
      <td>0.000</td>
      <td>3.098</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.065</td>
      <td>0.065</td>
      <td>0.000</td>
      <td>0.065</td>
      <td>0.070</td>
      <td>24.126</td>
    </tr>
  </tbody>
</table>


    ALGN



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
      <td>13.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.000</td>
      <td>-0.000</td>
      <td>-0.027</td>
      <td>0.015</td>
      <td>0.002</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.008</td>
      <td>0.008</td>
      <td>0.062</td>
      <td>0.044</td>
      <td>0.074</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.013</td>
      <td>-0.013</td>
      <td>-0.197</td>
      <td>0.000</td>
      <td>-0.197</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.013</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-8.261</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>10.543</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.021</td>
      <td>0.021</td>
      <td>0.000</td>
      <td>0.158</td>
      <td>0.158</td>
      <td>16.262</td>
    </tr>
  </tbody>
</table>


    AMD



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.001</td>
      <td>-0.001</td>
      <td>-0.005</td>
      <td>0.000</td>
      <td>-0.004</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.004</td>
      <td>0.004</td>
      <td>0.010</td>
      <td>0.001</td>
      <td>0.007</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.013</td>
      <td>-0.013</td>
      <td>-0.039</td>
      <td>0.000</td>
      <td>-0.024</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.004</td>
      <td>0.000</td>
      <td>-0.004</td>
      <td>-0.055</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.022</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.005</td>
      <td>0.000</td>
      <td>0.328</td>
    </tr>
  </tbody>
</table>


    AZO



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000</td>
      <td>6.000</td>
      <td>6.000</td>
      <td>6.000</td>
      <td>6.000</td>
      <td>6.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.026</td>
      <td>0.026</td>
      <td>-0.003</td>
      <td>0.040</td>
      <td>0.003</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.063</td>
      <td>0.063</td>
      <td>0.007</td>
      <td>0.062</td>
      <td>0.007</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-43.097</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.009</td>
      <td>0.000</td>
      <td>18.149</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.057</td>
      <td>0.000</td>
      <td>42.710</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.154</td>
      <td>0.154</td>
      <td>0.000</td>
      <td>0.154</td>
      <td>0.017</td>
      <td>73.614</td>
    </tr>
  </tbody>
</table>


    CL



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.002</td>
      <td>-0.002</td>
      <td>-0.012</td>
      <td>0.022</td>
      <td>0.016</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.012</td>
      <td>0.012</td>
      <td>0.014</td>
      <td>0.030</td>
      <td>0.026</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.026</td>
      <td>-0.026</td>
      <td>-0.039</td>
      <td>0.000</td>
      <td>-0.026</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.026</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-14.383</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.007</td>
      <td>0.007</td>
      <td>0.000</td>
      <td>-1.913</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.033</td>
      <td>0.026</td>
      <td>6.030</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.013</td>
      <td>0.013</td>
      <td>0.000</td>
      <td>0.092</td>
      <td>0.072</td>
      <td>26.771</td>
    </tr>
  </tbody>
</table>


    CTXS



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.003</td>
      <td>0.003</td>
      <td>-0.002</td>
      <td>0.002</td>
      <td>-0.001</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.006</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>0.004</td>
      <td>0.005</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.010</td>
      <td>0.000</td>
      <td>-0.010</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.004</td>
      <td>0.000</td>
      <td>-0.004</td>
      <td>-27.647</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-2.104</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.007</td>
      <td>0.007</td>
      <td>0.000</td>
      <td>0.005</td>
      <td>0.000</td>
      <td>4.733</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.014</td>
      <td>0.014</td>
      <td>0.000</td>
      <td>0.010</td>
      <td>0.010</td>
      <td>9.932</td>
    </tr>
  </tbody>
</table>


    HD



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.001</td>
      <td>0.001</td>
      <td>-0.011</td>
      <td>0.018</td>
      <td>0.013</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.022</td>
      <td>0.022</td>
      <td>0.014</td>
      <td>0.027</td>
      <td>0.056</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.032</td>
      <td>-0.032</td>
      <td>-0.036</td>
      <td>0.000</td>
      <td>-0.061</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.002</td>
      <td>-0.002</td>
      <td>-0.020</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-7.097</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.530</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.002</td>
      <td>0.002</td>
      <td>0.000</td>
      <td>0.032</td>
      <td>0.012</td>
      <td>4.019</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.062</td>
      <td>0.062</td>
      <td>0.000</td>
      <td>0.077</td>
      <td>0.155</td>
      <td>25.711</td>
    </tr>
  </tbody>
</table>


    KLAC



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.019</td>
      <td>-0.019</td>
      <td>-0.035</td>
      <td>0.008</td>
      <td>-0.003</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.080</td>
      <td>0.080</td>
      <td>0.055</td>
      <td>0.015</td>
      <td>0.052</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.198</td>
      <td>-0.198</td>
      <td>-0.198</td>
      <td>0.000</td>
      <td>-0.115</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.049</td>
      <td>-0.049</td>
      <td>-0.055</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.805</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.006</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.013</td>
      <td>0.000</td>
      <td>3.720</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.119</td>
      <td>0.119</td>
      <td>0.000</td>
      <td>0.044</td>
      <td>0.085</td>
      <td>14.324</td>
    </tr>
  </tbody>
</table>


    LRCX



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.023</td>
      <td>-0.023</td>
      <td>-0.040</td>
      <td>0.004</td>
      <td>-0.023</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.046</td>
      <td>0.046</td>
      <td>0.066</td>
      <td>0.010</td>
      <td>0.041</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.115</td>
      <td>-0.115</td>
      <td>-0.187</td>
      <td>0.000</td>
      <td>-0.128</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.081</td>
      <td>0.000</td>
      <td>-0.018</td>
      <td>-5.022</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.100</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.092</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.002</td>
      <td>0.002</td>
      <td>0.000</td>
      <td>0.034</td>
      <td>0.000</td>
      <td>8.986</td>
    </tr>
  </tbody>
</table>


    MAS



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16.000</td>
      <td>16.000</td>
      <td>16.000</td>
      <td>16.000</td>
      <td>16.000</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.017</td>
      <td>-0.017</td>
      <td>-0.024</td>
      <td>0.022</td>
      <td>0.016</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.044</td>
      <td>0.044</td>
      <td>0.025</td>
      <td>0.028</td>
      <td>0.040</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.144</td>
      <td>-0.144</td>
      <td>-0.082</td>
      <td>0.000</td>
      <td>-0.062</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.026</td>
      <td>-0.026</td>
      <td>-0.041</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-20.361</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.020</td>
      <td>0.015</td>
      <td>0.000</td>
      <td>1.822</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.033</td>
      <td>0.021</td>
      <td>8.208</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.049</td>
      <td>0.049</td>
      <td>0.000</td>
      <td>0.082</td>
      <td>0.102</td>
      <td>26.394</td>
    </tr>
  </tbody>
</table>


    ORLY



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.000</td>
      <td>7.000</td>
      <td>7.000</td>
      <td>7.000</td>
      <td>7.000</td>
      <td>7.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.021</td>
      <td>0.021</td>
      <td>0.000</td>
      <td>0.053</td>
      <td>0.030</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.047</td>
      <td>0.047</td>
      <td>0.000</td>
      <td>0.054</td>
      <td>0.052</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.011</td>
      <td>0.000</td>
      <td>-6.326</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.024</td>
      <td>0.000</td>
      <td>0.052</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.011</td>
      <td>0.011</td>
      <td>0.000</td>
      <td>0.102</td>
      <td>0.042</td>
      <td>12.820</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.126</td>
      <td>0.126</td>
      <td>0.000</td>
      <td>0.122</td>
      <td>0.122</td>
      <td>41.216</td>
    </tr>
  </tbody>
</table>


    ROK



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
      <td>14.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.011</td>
      <td>0.011</td>
      <td>-0.018</td>
      <td>0.022</td>
      <td>-0.024</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.033</td>
      <td>0.033</td>
      <td>0.031</td>
      <td>0.038</td>
      <td>0.052</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.008</td>
      <td>-0.008</td>
      <td>-0.087</td>
      <td>0.000</td>
      <td>-0.108</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.018</td>
      <td>0.000</td>
      <td>-0.077</td>
      <td>-9.114</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.971</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.026</td>
      <td>0.000</td>
      <td>4.999</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.105</td>
      <td>0.105</td>
      <td>0.000</td>
      <td>0.105</td>
      <td>0.065</td>
      <td>29.213</td>
    </tr>
  </tbody>
</table>


    TER



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12.000</td>
      <td>12.000</td>
      <td>12.000</td>
      <td>12.000</td>
      <td>12.000</td>
      <td>12.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.006</td>
      <td>0.006</td>
      <td>-0.009</td>
      <td>0.026</td>
      <td>0.007</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.039</td>
      <td>0.039</td>
      <td>0.015</td>
      <td>0.069</td>
      <td>0.079</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.048</td>
      <td>-0.048</td>
      <td>-0.036</td>
      <td>0.000</td>
      <td>-0.084</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.007</td>
      <td>-0.007</td>
      <td>-0.019</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-3.719</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.010</td>
      <td>0.010</td>
      <td>0.000</td>
      <td>0.007</td>
      <td>0.000</td>
      <td>1.190</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.095</td>
      <td>0.095</td>
      <td>0.000</td>
      <td>0.239</td>
      <td>0.239</td>
      <td>10.465</td>
    </tr>
  </tbody>
</table>


    TXN



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.019</td>
      <td>-0.019</td>
      <td>-0.017</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.041</td>
      <td>0.041</td>
      <td>0.039</td>
      <td>0.013</td>
      <td>0.016</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.153</td>
      <td>-0.153</td>
      <td>-0.153</td>
      <td>0.000</td>
      <td>-0.006</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.028</td>
      <td>-0.028</td>
      <td>-0.019</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-5.138</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.003</td>
      <td>0.000</td>
      <td>-0.778</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.016</td>
      <td>0.009</td>
      <td>1.969</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.021</td>
      <td>0.021</td>
      <td>0.000</td>
      <td>0.042</td>
      <td>0.054</td>
      <td>16.226</td>
    </tr>
  </tbody>
</table>


    VRSN



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16.000</td>
      <td>16.000</td>
      <td>16.000</td>
      <td>16.000</td>
      <td>16.000</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.006</td>
      <td>0.006</td>
      <td>-0.011</td>
      <td>0.045</td>
      <td>0.019</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.056</td>
      <td>0.056</td>
      <td>0.026</td>
      <td>0.081</td>
      <td>0.058</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.100</td>
      <td>-0.100</td>
      <td>-0.089</td>
      <td>0.000</td>
      <td>-0.017</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.006</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-7.556</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.019</td>
      <td>0.019</td>
      <td>0.000</td>
      <td>0.044</td>
      <td>0.001</td>
      <td>2.175</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.150</td>
      <td>0.150</td>
      <td>0.000</td>
      <td>0.233</td>
      <td>0.219</td>
      <td>33.636</td>
    </tr>
  </tbody>
</table>


    WAT



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11.000</td>
      <td>11.000</td>
      <td>11.000</td>
      <td>11.000</td>
      <td>11.000</td>
      <td>11.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.011</td>
      <td>0.011</td>
      <td>-0.006</td>
      <td>0.013</td>
      <td>-0.005</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.038</td>
      <td>0.038</td>
      <td>0.020</td>
      <td>0.037</td>
      <td>0.021</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.003</td>
      <td>-0.003</td>
      <td>-0.066</td>
      <td>0.000</td>
      <td>-0.066</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-259.686</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-4.217</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>18.359</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.125</td>
      <td>0.125</td>
      <td>0.000</td>
      <td>0.122</td>
      <td>0.016</td>
      <td>46.950</td>
    </tr>
  </tbody>
</table>


    YUM



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_adj_close</th>
      <th>%_close</th>
      <th>%_high</th>
      <th>%_low</th>
      <th>%_open</th>
      <th>%_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
      <td>15.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.007</td>
      <td>-0.007</td>
      <td>-0.009</td>
      <td>0.010</td>
      <td>0.003</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.020</td>
      <td>0.020</td>
      <td>0.013</td>
      <td>0.019</td>
      <td>0.019</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.069</td>
      <td>-0.069</td>
      <td>-0.035</td>
      <td>0.000</td>
      <td>-0.030</td>
      <td>-inf</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.017</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-3.364</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.299</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.013</td>
      <td>0.004</td>
      <td>1.714</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.004</td>
      <td>0.004</td>
      <td>0.000</td>
      <td>0.052</td>
      <td>0.061</td>
      <td>9.637</td>
    </tr>
  </tbody>
</table>






### Conclusions

Based on the above data we can conclude:

- After market closing, historical intraday data is slightly different than during live session
- 15 out of around 300 one-minute bars have a price difference of less than 0.2%
- Overall, price differences exist do to adjustments after market closing, but are small



