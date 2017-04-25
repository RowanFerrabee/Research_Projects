
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils


print('Starting')

(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_classes = 10
num_channels = 1
width = int(X_train.shape[1])
height = int(X_train.shape[2])

print('Got data')

# Get data into shape and type keras can use
X_train = X_train.reshape(X_train.shape[0], num_channels, width, height).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_channels, width, height).astype('float32')
y_train = np_utils.to_categorical(y_train, num_classes) 
y_test = np_utils.to_categorical(y_test, num_classes)

# Normalize data
X_train /= 255
X_test /= 255

print('Data formatted and normalized')

model = Sequential()
# CNN Layers
model.add(Convolution2D(12,(3,3),activation='relu',input_shape=(num_channels,width,height)))
model.add(Convolution2D(12,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(12,(7,7),activation='relu'))
model.add(Dropout(0.25))
# FC Layers
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

print('Model compiled')

model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1)

print('Done fitting')

model_json = model.to_json()
with open('model5x5.json', 'w') as json_file:
	json_file.write(model_json)
model.save_weights('trained_model5x5.h5')

score = model.evaluate(X_test, y_test, verbose=0)

print('Done evaluation. Score: {0}'.format(score))