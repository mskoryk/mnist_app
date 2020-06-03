import streamlit as st

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
from keras import losses


import numpy as np
from sklearn.preprocessing import MinMaxScaler

num_classes = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# def get_model():  
#     K.clear_session()
#     inp = Input(shape=(28,28,1))  #2D matrix of n MFCC bands by m audio length.
#     x = Conv2D(32, (3,3), padding="same")(inp)    
#     x = Activation("relu")(x)
#     x = Conv2D(64, (3,3), padding="same")(x)    
#     x = Activation("relu")(x)

#     x = MaxPooling2D(pool_size=(2,2))(x)
#     x = Dropout(rate=0.25)(x)    

#     x = Flatten()(x)
#     x = Dense(128)(x) 
#     x = Activation("relu")(x)
#     x = Dropout(rate=0.5)(x)    
#     out = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=inp, outputs=out)    
    
#     model.compile(optimizer=keras.optimizers.Adadelta(), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    
#     model.load_weights('./mnist.h5')
#     return model

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    model.load_weights('./mnist.h5')
    return model

def mnist_predict(image):    
    image = image.astype(np.float32)
    
    scaler = MinMaxScaler()
    original_shape = image.shape
    image = scaler.fit_transform(image.reshape(-1, 1))
    image = image.reshape(original_shape)
    
    model = get_model()
    preds = np.argmax(model.predict(image.reshape(1, img_rows, img_cols, 1)))
    return preds