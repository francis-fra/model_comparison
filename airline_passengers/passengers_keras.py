# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def load_data():
    data_folder = '/home/fra/DataMart/datacentre/opendata/time_series/airline-passengers.csv'
    # load the dataset
    dataframe = pandas.read_csv(data_folder, usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset

def transform(dataset):

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    return (train, test, scaler)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    # x is a delayed vector of y
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def reshape_data(df):
    "reshape input to be [samples, time steps, features]"
    return np.reshape(df, (df.shape[0], 1, df.shape[1]))

def get_train_test_data(look_back):
    df = load_data()
    train_df, test_df, scaler = transform(df)
    x_train, y_train = create_dataset(train_df, look_back)
    x_test, y_test = create_dataset(test_df, look_back)
    return (x_train, x_test, y_train, y_test, scaler)


def build_model(look_back):
    model = Sequential()
    model.add(LSTM(4)) 
    model.add(Dense(1))
    return model

if __name__ == '__main__':
    look_back = 1

    (x_train, x_test, y_train, y_test, scaler) =  get_train_test_data(look_back)
    # for LSTM inputs
    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)

    model = build_model(look_back) 

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

    # make predictions
    pred = model.predict(x_test)

    # un-normalize time series
    pred = scaler.inverse_transform(pred)
    actual = scaler.inverse_transform(y_test[:, np.newaxis])

    # calculate root mean squared error
    testScore = math.sqrt(mean_squared_error(actual, pred))
    print('Test Score: %.2f RMSE' % (testScore))

