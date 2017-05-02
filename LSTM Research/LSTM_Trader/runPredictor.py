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
from keras.models import model_from_json


date_1 = '2016-01-01'
date_2 = '2017-04-26'
stock_name = "YHOO"

numpy.random.seed(7)

#Process financial data into a series
dataset = process_stock_data(date_1, date_2, stock_name)

#Map scalars from 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# #Split data into testing and training
# train_size = int(len(dataset) * 0.67) - 1
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# print len(train), len(test) 

# Create dataset using specified lookback
look_back = 10
testX, testY = create_dataset(dataset, look_back)

# Reshape input to be [samples, time steps, features]
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# load json and create model
json_file = open('Models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("Models/model.h5")
print("Loaded model from disk")

stock_name = "YHOO"

testPredict = model.predict(testX)

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

print testPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[(look_back*2):len(dataset) - 1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(testPredictPlot)
plt.legend(['Actual', 'Projected'])
plt.show()




