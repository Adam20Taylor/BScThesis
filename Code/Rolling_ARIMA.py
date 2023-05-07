from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from numpy import ndarray
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import time
# load dataset

use = ['Passengers']
series = read_csv('Data/airline-passengers.csv', header=0, usecols=use)

"""
use = ['World']
series = read_csv('Data/total_cases.csv', header=0, usecols=use)
"""
#series = read_csv('Data/temperature.csv')
"""
colnames = ['date','time','temp','redu']
use = ['temp']
series = read_csv('Data/smhi_data_gothenburg_cutdown.csv',sep=';',header=0,names=colnames,usecols=use)
"""
#series.index = series.index.to_period('M')
# split into train and test sets
start_time = time.time()
X = series.values
size = int(len(X) * 0.66)
# order p=3, d=0 for both temperature datasets / d=1 for airline passengers and total cases, q=3
order = (20,1,20)
max_order = max(order)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history[len(history)-max_order:], order=order, enforce_stationarity=False)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('t=%f, predicted=%f, expected=%f' % (t, yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
mae = mean_absolute_error(test,predictions)
mape = mean_absolute_percentage_error(test,predictions)
print('Test RMSE: %.3f, Test MAE: %.3f, Test MAPE: %.3f' % (rmse,mae,mape))
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
print("--- %s seconds ---" % (time.time() - start_time))