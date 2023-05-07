import tensorflow as tf
import os
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
import time

df = pd.read_csv('data/airline-passengers.csv')
timeseries = df[["Passengers"]].values.astype('float32')

"""
cols = ['World']
df = pd.read_csv('Data/total_cases.csv',header=0, usecols=cols)
df = df.dropna()
df = df/10000000
timeseries = df[["World"]].values.astype('float32')
"""
"""
df = pd.read_csv('data/temperature.csv', names=['Temp'])
timeseries = df[["Temp"]].values.astype('float32')
"""
"""
names = ['Date','Time','Temp','Grade']
use = ['Temp']
df = pd.read_csv('Data/smhi_data_gothenburg_cutdown.csv', sep=';',header=None,names=names, usecols=use)
timeseries = df[["Temp"]].values.astype('float32')
"""


start_time = time.time()

train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

# Number of epochs, change with size of dataset
n_epochs = 3000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation, change with size of dataset
    if epoch % 300 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        test_actuals = y_test.resize(len(test) - lookback, lookback)
        test_predictions = y_pred.resize(len(test) - lookback, lookback)
        test_mae = mean_absolute_error(test_actuals, test_predictions)
        test_mape = mean_absolute_percentage_error(test_actuals, test_predictions)
    print("Epoch %d: test RMSE %.4f, test MAE: %.4f, test MAPE: %.4f, time %.4f" % (epoch, test_rmse,test_mae,test_mape,(time.time() - start_time)))

    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        #train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size + lookback:len(timeseries)] = model(X_test)[:, -1, :]
    # plot
    #plt.plot(timeseries, c='b')
    #plt.plot(train_plot, c='r')
    plt.title('GRU epoch: %i' % epoch)
    plt.plot(range(train_size, train_size+len(test)),test)
    plt.plot(test_plot, c='g')
    plt.show()