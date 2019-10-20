from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, Flatten
# import two new layers, a 2D layer and one that flattens its input to vectors

np.random.seed(123)
X = np.load('../generated_games/features-200.npy')
Y = np.load('../generated_games/labels-200.npy')

samples = X.shape[0]
size = 9
input_shape = (size, size, 1)  # The input data shape is 3-dimensional, we use one plane of a 9x9 board representation

X = X.reshape(samples, size, size, 1)  # reshape our input data accordingly

train_samples = 10000
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

model = Sequential()
model.add(Conv2D(filters=32,  # The first layer in our network is a Conv2D layer with 32 output filters
                 kernel_size=(3, 3),  # For this layer we choose a 3 by 3 convolutional kernel
                 activation='sigmoid',
                 input_shape=input_shape))
# The second layer in another convolution. leave out the "filters" and "kernel_size" arguments for brevity
model.add(Conv2D(64, (3, 3), activation='sigmoid'))

model.add(Flatten())  # then flatten the 3D output of the previous convolutional layer...

model.add(Dense(128, activation='sigmoid'))
model.add(Dense(size * size, activation='sigmoid'))  # ... and follow up with two more dense layers
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=64,
          epochs=5,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
