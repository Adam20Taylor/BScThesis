from pandas import read_csv
from numpy import array
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import time

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

"""
use = ['Passengers']
df = read_csv('Data/airline-passengers.csv', header=0, usecols=use)
"""

df = read_csv('Data/total_cases.csv', header=0, usecols=['World'])
#df = read_csv('Data/temperature.csv', header=None)
"""
use = ['temp']
cols = ['date','time','temp','grade']
df = read_csv('Data/smhi_data_gothenburg_cutdown.csv',sep=';', header=None, names=cols , usecols=use)
"""
df = df.dropna()
start_time = time.time()
size = int(len(df) * 0.66)
seq, test = df[0:size], df[size:]
n_steps_in, n_steps_out = 3, 1
X, y = split_sequence(seq, n_steps_in, n_steps_out)
test_X, test_y = split_sequence(test,n_steps_in,n_steps_out)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))
model = Sequential()
model.add(GRU(100, activation='relu', return_sequences=True, input_shape=(n_steps_in,
n_features)))
model.add(GRU(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=1)
weights = model.get_weights()
predictions = list()
for t in range(len(test)-n_steps_in):
    x_input = test_X[t]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    predictions.append(yhat)
    if t % 300 == 0:
        print(t)
predictions = [float(x[0]) for x in predictions]
RMSE = sqrt(mean_squared_error(test[n_steps_in:], predictions))
MAE = mean_absolute_error(test[n_steps_in:], predictions)
MAPE = mean_absolute_percentage_error(test[n_steps_in:], predictions)
plt.plot(test.index[n_steps_in:], predictions)
plt.plot(test)
plt.show()
print("RMSE: %f" % RMSE)
print("MAE: %f" % MAE)
print("MAPE: %f" % MAPE)
print("--- %s seconds ---" % (time.time() - start_time))