from pandas import read_csv
from matplotlib import pyplot
from numpy import ndarray
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import time
# load dataset
"""
use = ['Passengers']
series = read_csv('Data/airline-passengers.csv', header=0, usecols=use)
order = (1,1,1)
seasonal_order = (1,1,1,12)
"""

use = ['World']
series = read_csv('Data/total_cases.csv', header=0, usecols=use)
series = series.dropna()
series = series/10000000
order = (4,1,2)
seasonal_order = (1,1,1,7)
"""
series = read_csv('Data/temperature.csv')
order = (20,0,10)
seasonal_order = (0,0,0,0)
"""
"""
colnames = ['date','time','temp','redu']
use = ['temp']
series = read_csv('Data/smhi_data_gothenburg_cutdown.csv',sep=';',header=0,names=colnames,usecols=use)
order = (20,0,0)
seasonal_order = (1,0,0,24)
"""
#series.index = series.index.to_period('M')
# split into train and test sets
start_time = time.time()
X = series.values
size = int(len(X) * 0.66)
max_order = max(order)
train, test = X[0:size], X[size:]
# cut-down size for machine temperature dataset
#train, test = X[0:5000], X[5000:8000]
# cut-down size for SMHI temperature dataset
#train, test = X[0:20000], X[20000:25000]
model = SARIMAX(train, order=order,seasonal_order=seasonal_order)
model_fit = model.fit()
predictions = model_fit.forecast(len(test))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
mae = mean_absolute_error(test,predictions)
mape = mean_absolute_percentage_error(test,predictions)
print('Test RMSE: %.3f, Test MAE: %.3f, Test MAPE: %.3f' % (rmse,mae,mape))
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.ylabel(order + seasonal_order)
pyplot.show()
print("--- %s seconds ---" % (time.time() - start_time))