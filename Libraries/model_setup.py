"""Specifies model architecture"""

import tensorflow as tf
from tensorflow import keras

def getModelA():
    #model architecture
    model = keras.Sequential([keras.layers.Dense(500, activation= tf.nn.sigmoid),
                              keras.layers.Dense(200, activation= tf.nn.sigmoid),
                              keras.layers.Dense(2,   activation= tf.nn.softmax)])

    #specifies optimizer and lossfunctions
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

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