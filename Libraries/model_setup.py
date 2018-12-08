"""Specifies model architecture"""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Dropout
from keras import backend as K


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


def distributed_into_one(input_shape):
    # model architecture
    m = Sequential()
    # m.add(Masking(mask_value=0., input_shape=(InputDataSet.shape[1], InputDataSet.shape[2])))
    m.add(LSTM(8, return_sequences=True, input_shape=input_shape))
    m.add(LSTM(16, return_sequences=True))
    m.add(TimeDistributed(Dense(1, activation='sigmoid')))
    m.add(Lambda(lambda x: K.max(x, keepdims=True)))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


def singleLabel_1(input_shape):
    # model architecture
    m = Sequential()
    m.add(LSTM(8, return_sequences=True, input_shape=input_shape))
    m.add(Flatten())
    m.add(Dense(1, activation='sigmoid'))
    # specifies optimizer and lossfunctions
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


def singleLabel_2(input_shape):
    # model architecture
    m = Sequential()
    m.add(LSTM(8, return_sequences=True, input_shape=input_shape))
    m.add(LSTM(16, return_sequences=True))
    m.add(Flatten())
    m.add(Dense(1, activation='sigmoid'))
    # specifies optimizer and lossfunctions
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


def singleLabel_3(input_shape):
    # model architecture
    m = Sequential()
    m.add(LSTM(8, return_sequences=True, input_shape=input_shape))
    m.add(LSTM(16, return_sequences=True))
    m.add(LSTM(8, return_sequences=True))
    m.add(Flatten())
    m.add(Dense(1, activation='sigmoid'))
    # specifies optimizer and lossfunctions
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


def singleLabel_HighNumber(input_shape):
    # model architecture
    m = Sequential()
    # m.add(Masking(mask_value=0., input_shape=(InputDataSet.shape[1], InputDataSet.shape[2])))
    m.add(LSTM(8, return_sequences=True, input_shape=input_shape, recurrent_dropout=0.2))
    m.add(Dropout(0.2))
    m.add(LSTM(50, return_sequences=True, recurrent_dropout=0.2))
    m.add(Dropout(0.2))
    m.add(LSTM(16, return_sequences=True, recurrent_dropout=0.2))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(1, activation='sigmoid'))
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m



modelDict = {
    'singleLabel_1': singleLabel_1,
    'singleLabel_2': singleLabel_2,
    'singleLabel_3': singleLabel_3,
    'singleLabel_HighNUmber': singleLabel_HighNumber
}