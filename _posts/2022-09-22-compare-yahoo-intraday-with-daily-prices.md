# Study comparing Yahoo Intraday with Daily Closing Price Data

Author: Deeepwin  
Date: 17.09.2022 
***

## Background

There is not much information on the internet in how Yahoo daily closing prices are calculated and how they compare to the intraday data. At the NYSE daily closing price is determine by an [closing auction] (https://www.nyse.com/article/nyse-closing-auction-insiders-guide). In this article we make an attempt to compare real data. The goal ist to compare:

1) The day end prices of historical intraday data such as 1m, 5m and 30m data to the daily closing prices
2) Compare live intraday data with the historical intraday data
 

## Imports & Settings


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

data_from_date  = datetime.datetime.now() - pd.offsets.Day(2) # Yahoo allows to download last 30 days only
data_end_date   = datetime.datetime.now() - pd.offsets.BDay(2)
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    /home/martin/GitHub


## Load Dates to Test

Define the tickers you want to compare.


```python

tickers = ['ALGN', 'AMD']

metric = 'adj_close'
```


```python
df_dates = yf.download([tickers[0]], interval='1D', start=data_from_date, end=data_end_date, progress=False)
dates = df_dates.index
```

# Download Prices


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

# Analyse Result

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
      <td>18.000000</td>
      <td>18.000000</td>
      <td>18.000000</td>
      <td>18.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.008333</td>
      <td>-0.008333</td>
      <td>-0.008333</td>
      <td>0.007556</td>
      <td>-0.009111</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.025474</td>
      <td>0.025474</td>
      <td>0.025474</td>
      <td>0.089016</td>
      <td>0.025621</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.039000</td>
      <td>-0.039000</td>
      <td>-0.039000</td>
      <td>-0.138000</td>
      <td>-0.039000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.019500</td>
      <td>-0.019500</td>
      <td>-0.019500</td>
      <td>-0.048000</td>
      <td>-0.019500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.012500</td>
      <td>-0.012500</td>
      <td>-0.012500</td>
      <td>0.005500</td>
      <td>-0.012500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.053250</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.078000</td>
      <td>0.078000</td>
      <td>0.078000</td>
      <td>0.206000</td>
      <td>0.078000</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusions

We can make the following conclusions:

- Worst price difference is -0.73% to daily intraday over last 30 trading days.
- Alpaca standard deviation is better with 0.036 compared to Yahoo with 0.13.
- VWAP with 1m bars between 15:30 and 16:00 is worse.
- Overall, price difference appear not to large.


## Compare Live Intraday Data with Historical

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

    Stored 'df_live_data' (DataFrame)
    Stored 'test_date' (Timestamp)


Data is stored to Jupyter Notebook. Afterwards we have to wait for one day to continue to download same day as historical dataset.


```python
%store df_live_data test_date
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

    # get previously stored data for that ticker
    df_y_daily.filter(regex='^' + date.date().strftime('%Y-%m-%d'), axis=0)

    df_live_ticker = df_y_1m_live.filter(regex=ticker, axis=0).reset_index().drop('ticker', axis=1).set_index('date').sort_index(axis=1)

    # download historical data for same date
    start_date = (test_date.date() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    end_date   = (test_date.date() + pd.offsets.BDay(1)).strftime('%Y-%m-%d')

    df_y_1m = yf.download([ticker], interval='1m', start=date, end=end_date, progress=False)
    df_y_1m.columns = map(str.lower, df_y_1m.columns)
    df_y_1m.rename(columns={'adj close': 'adj_close'}, inplace=True)
    df_y_1m.index.names = ['date']
    df_y_1m = df_y_1m.filter(regex='^' + test_date.date().strftime('%Y-%m-%d'), axis=0)
    df_y_1m.sort_index(inplace=True)
    df_y_1m.sort_index(inplace=True, axis=1)
    
    # test
    #df_y_1m.at[df_y_1m.index[5], 'low'] = 39
    
    print(ticker)
    display(HTML(df_y_1m.iloc[:nb_of_lines].compare(df_live_ticker.iloc[:nb_of_lines]).to_html()))
```

    ALGN



<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">low</th>
    </tr>
    <tr>
      <th></th>
      <th>self</th>
      <th>other</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-09-21 09:35:00-04:00</th>
      <td>39.000</td>
      <td>232.200</td>
    </tr>
  </tbody>
</table>


    AMD



<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">low</th>
    </tr>
    <tr>
      <th></th>
      <th>self</th>
      <th>other</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-09-21 09:35:00-04:00</th>
      <td>39.000</td>
      <td>74.851</td>
    </tr>
  </tbody>
</table>



```python

```
