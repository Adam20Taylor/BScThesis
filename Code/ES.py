import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

"""
use = ['Passengers']
df = pd.read_csv('Data/airline-passengers.csv', header=0, usecols=use)
"""

use = ['date','World']
df = pd.read_csv('Data/total_cases.csv', header=0, usecols=use, parse_dates=['date'],index_col=0)
df = df.dropna()
df = df/10000000

"""
df = pd.read_csv('Data/temperature.csv')
df.index = pd.date_range('20230414 12:00:00',freq='5min',periods=len(df))
"""
"""
colnames = ['date','time','temp','redu']
use = ['temp']
df = pd.read_csv('Data/smhi_data_gothenburg_cutdown.csv',sep=';',header=0,names=colnames,usecols=use)
"""



start_time = time.time()
size = int(len(df) * 0.66)
train, test = df[0:size], df[size:]
#train, test = df[0:5000], df[5000:8000]
#train, test = df[0:15000], df[15000:19000]

rangesize = 3
# Exponential smoothening for dataset 1
#model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
# Exponential smoothening for dataset 2
model = ExponentialSmoothing(train, trend='add').fit()

#model = SimpleExpSmoothing(train).fit()
# Exponential smoothening for dataset 4
#model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=24).fit()

predictions = model.forecast(len(test))
RMSE = sqrt(mean_squared_error(test, predictions))
MAE = mean_absolute_error(test, predictions)
MAPE = mean_absolute_percentage_error(test, predictions)
print("RMSE: %f" % RMSE)
print("MAE: %f" % MAE)
print("MAPE: %f" % MAPE)
print("--- %s seconds ---" % (time.time() - start_time))

#plt.plot(train, label='Actual')
plt.plot(test, label='test')
#plt.plot(ts_es1, label='es 1')
#plt.plot(ts_es2, label='es 2')
plt.plot(predictions, label='es 3')
plt.legend(loc='best')
plt.show()
