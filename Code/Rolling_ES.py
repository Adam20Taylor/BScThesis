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

df = pd.read_csv('data/airline-passengers.csv', parse_dates=['Month'], index_col='Month')
#use = ['World']
#df = pd.read_csv('Data/total_cases.csv', header=0, usecols=use)

#df = pd.read_csv('Data/temperature.csv')
"""
colnames = ['date','time','temp','redu']
use = ['temp']
df = pd.read_csv('Data/smhi_data_gothenburg_cutdown.csv',sep=';',header=0,names=colnames,usecols=use)
"""


#df.index.freq = 'MS'
df = df.dropna()
X = df.values
start_time = time.time()
size = int(len(df) * 0.66)
train, test = X[0:size], X[size:]
history = [x for x in train]
rangesize = 12
#es3 = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12).fit()

predictions = list()
for t in range(len(test)):
    es = SimpleExpSmoothing(history[len(history)-rangesize:]).fit()
    out = es.forecast(1)
    yhat = out[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    if t % 1000 == 0:
        print(t)

#predictions = es3.forecast(len(test))
RMSE = sqrt(mean_squared_error(test, predictions))
MAE = mean_absolute_error(test, predictions)
MAPE = mean_absolute_percentage_error(test, predictions)
print("RMSE: %f" % RMSE)
print("MAE: %f" % MAE)
print("MAPE: %f" % MAPE)
print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(test, label='test')
#plt.plot(ts_es1, label='es 1')
#plt.plot(ts_es2, label='es 2')
plt.plot(predictions, label='es 3')
plt.legend(loc='best')
plt.show()
