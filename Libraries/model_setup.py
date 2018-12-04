"""Specifies model architecture"""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense

def distributed_label(input_shape):
    #model architecture
    m = Sequential()
    # m.add(Masking(mask_value=0., input_shape=(InputDataSet.shape[1], InputDataSet.shape[2])))
    m.add(LSTM(8, return_sequences=True, input_shape=input_shape))
    m.add(LSTM(16, return_sequences=True))
    m.add(LSTM(8, return_sequences=True))
    m.add(TimeDistributed(Dense(1, activation='sigmoid')))
    # m.add(TimeDistributed(Flatten()))
    # m.add(Flatten())
    #m.add(Dense(1, activation='sigmoid'))
    #specifies optimizer and lossfunctions
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def getModelB():
    #model architecture
    model = keras.Sequential([keras.layers.Dense(100, activation=tf.nn.sigmoid),
                              keras.layers.Dense(50, activation = tf.nn.sigmoid),
                              keras.layers.Dense(2,  activation=tf.nn.softmax)])

    #specifies optimizer and lossfunctions
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def getModelC():
    #model architecture
    model = keras.Sequential([keras.layers.Dense(50, activation=tf.nn.sigmoid),
                              keras.layers.Dense(20, activation = tf.nn.sigmoid),
                              keras.layers.Dense(2,  activation=tf.nn.softmax)])

    #specifies optimizer and lossfunctions
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model