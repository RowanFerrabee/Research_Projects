from helpers import *
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import dill
import pickle

date_1 = '2017-01-17'
date_2 = '2017-03-17'
stock_name = "YHOO"

#Process financial data into a series
dataset = process_stock_data(date_1, date_2, stock_name)
#plt.plot(dataset)
#plt.show()

#Map scalars from 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#Split data into testing and training
train_size = int(len(dataset) * 0.67) - 1
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print len(train), len(test) 

#Create dataset using specified lookback
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Create model architecture
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

#Save model for evaluation
pickle.dump(model, open("Models/" + stock_name + ".bin", "wb"))








