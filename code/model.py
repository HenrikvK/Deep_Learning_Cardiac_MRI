from __future__ import print_function
import numpy as np
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Flatten, Activation, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)


def get_model_reference():
    model = Sequential()
    model.add(Activation(activation=center_normalize, input_shape=(30, 64, 64)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, W_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=root_mean_squared_error)
    return model


def get_model():
    # image input layer
    img = Input(shape=(30,64,64))
    img_act = Activation(activation=center_normalize)(img)

    # metadata input layer for scaling data
    meta = Input(shape=(2,))

    # CNN layer groups
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(img_act)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='valid')(conv1)
    zerop1 = ZeroPadding2D(padding=(1,1))(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(zerop1)
    drop1 = Dropout(0.25)(pool1)

    conv3 = Conv2D(96, (3,3), activation='relu', padding='same')(drop1)
    conv4 = Conv2D(96, (3,3), activation='relu', padding='valid')(conv3)
    zerop2 = ZeroPadding2D(padding=(1,1))(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(zerop2)
    drop2 = Dropout(0.25)(pool2)

    conv5 = Conv2D(128, (3,3), activation='relu', padding='same')(drop2)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv6)
    drop3 = Dropout(0.25)(pool3)

    flat = Flatten()(drop3)

    # merge layers to the fully connected
    merge = concatenate([flat, meta])

    dense = Dense(1024, W_regularizer=l2(1e-3), activation='relu')(merge)
    drop4 = Dropout(0.5)(dense)

    # prediction output
    output = Dense(1)(drop4)
    model = Model(inputs=[img, meta], outputs=output)

    adam = Adam(lr=0.0005, clipvalue=0.5) # not tuned - potential improvement
    model.compile(optimizer=adam, loss=root_mean_squared_error)
    return model