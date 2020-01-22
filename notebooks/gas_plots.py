from matplotlib import pyplot as plt
import seaborn as sns
import gzip
import json
import datetime
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from statsmodels.graphics.tsaplots import plot_acf

with gzip.open("./data/gas-prices-with-txs-3months.jsonl.gz") as f:
    data = []
    for i, line in enumerate(f):
        data.append(json.loads(line))
        if i >= 50_000:
            break
    df = json_normalize(data)
    df['datetime'] = pd.to_datetime(df['timestamp'],unit='s')

# add column for % of gas utilization
df['gas_utilization'] = df['gas_used']/df['gas_limit']*100

print(df.columns)

plt.style.use('ggplot')
df.set_index('datetime', inplace=True)

# Func: compute savings for a 50k gas costs per block for min, max and avg prices
def compute_savings():
    avg_prices = df.resample('3H')['average_gas_price'].mean() * 50_000
    min_prices = df.resample('3H')['min_price_tx.gas_price'].mean() * 50_000
    max_prices = df.resample('3H')['max_price_tx.gas_price'].mean() * 50_000
    
    print(sum(avg_prices)/(1e18))
    print(sum(min_prices)/(1e18))
    print(sum(max_prices)/(1e18))


# Plot 1) Max - avg - min gas price over time
def plot_avg_price_spread():
    #df.set_index('datetime', inplace=True)
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    df.resample('3H')['average_gas_price'].mean().plot(ax=ax, label="Average")
    df.resample('3H')['min_price_tx.gas_price'].mean().plot(ax=ax, label="Minimum")
    df.resample('3H')['max_price_tx.gas_price'].mean().plot(ax=ax, label="Maximum")
    ax.set_yscale('log')
    ax.set_ylabel('Gas Price (wei)')
    ax.set_xlabel('Time (3 Hour Averages)')
    ax.legend()

    plt.savefig('gas_price_avg_spread.png', format='png', dpi=100)
    plt.show()

#compute_savings()

def remove_outliers(data, columns):
    for col in columns:
        if col not in data:
            continue
        series = data[col]
        mean = series.mean()
        std_dev = series.std()
        data = data[(series-mean).abs() <= 3*std_dev]
    return data

def min_max_scale(data, columns):
    for col in columns:
        if col not in data:
            continue
        min_s = data[col].min()
        max_s = data[col].max()
        data[col] = (data[col] - min_s) / (max_s - min_s)
    return data

# Variables of interest
cols = ['tx_count', 'gas_utilization', 'average_gas_price', 'min_price_tx.gas_price', 'max_price_tx.gas_price']


# Plot all the variables
df = remove_outliers(df, cols)
df_scale = min_max_scale(df, cols)
df_resample = df_scale.resample('10T').mean()
fig = plt.figure(figsize = (14, 8))
for col in cols:
    # print(col, "\n")
    plt.plot(df_resample[col], label=col)
plt.legend()
plt.show()

# Compute cross correlation without lags between all variables
df_cross_corr = df[cols]
fig = plt.figure(figsize= (14, 8))
plt.matshow(df_cross_corr.corr())
plt.ylim(-.5, 4.5)
plt.xticks(np.arange(5), cols, rotation=90)
plt.yticks(np.arange(5), cols, rotation=0)
cb = plt.colorbar()
plt.show()


# Compute cros-correlation with lags
df_lags = pd.DataFrame()
df_cross_corr = df[cols]
#reample
df_cross_corr = df_cross_corr.resample('10T').mean()

lags = [1, 2, 3, 4, 5, 10, 48*3+1]

cc = df_cross_corr.corr()
df_lags['lagged_0'] = cc['average_gas_price'] 

for lag in lags:
    df_temp = df_cross_corr.copy()
    df_temp['average_gas_price'] = df_temp['average_gas_price'].shift(lag)
    df_temp = df_temp[lag:]
    cc = df_temp.corr()
    # print(df_temp['tx_count'][:5])
    df_lags['lagged_' + str(lag)] = cc['average_gas_price'] 

print(df_lags.head())
fig = plt.figure(figsize= (14, 8))
plt.matshow(df_lags)
plt.xticks(np.arange(8), df_lags.columns, rotation=90)
plt.yticks(np.arange(5), cols, rotation=0)
cb = plt.colorbar()
plt.show()


# Compute autocorrelations
for col in cols:
    print(col, "\n")
    # fig, ax = plt.subplots(figsize = (14, 8))
    plot_acf(df[col], lags = 7000)
    # ax.acorr(df_resample[col], maxlags=50, label=col)
    # plt.legend()
    plt.show()