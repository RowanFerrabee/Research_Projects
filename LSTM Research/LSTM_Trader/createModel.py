from helpers import *
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

date_1 = '2012-01-17'
date_2 = '2016-01-01'
stock_name = "YHOO"

numpy.random.seed(7)

#Process financial data into a series
dataset = process_stock_data(date_1, date_2, stock_name)

#Map scalars from 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(dataset)

#Create dataset using specified lookback
look_back = 10
trainX, trainY = create_dataset(train, look_back)

print trainX
print "-----"
print trainY

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

#Create model architecture
model = Sequential()
model.add(LSTM(5, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(5, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# serialize model to JSON
model_json = model.to_json()
with open("Models/model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("Models/model.h5")
print("Saved model to disk")








