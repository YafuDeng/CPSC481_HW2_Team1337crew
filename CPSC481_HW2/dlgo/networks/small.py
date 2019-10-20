from __future__ import absolute_import

from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        # use zero padding layers to enlarge input images.
        ZeroPadding2D(padding=3, input_shape=input_shape, data_format='channels_first'),
        Conv2D(48, (7, 7), data_format='channels_first'),
        Activation('relu'),
        # By using `channels_first` we specify that the input plane dimension for our features comes first
        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),
    ]
