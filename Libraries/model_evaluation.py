"""Contains Tools for Evaluating the Training Procedue and Helper Functions for Debugging and Documentation purpose"""
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import numpy as np
from matplotlib2tikz import save as tikz_save

def calculateConfusionMatrix(labels, predictions):
    confusion = tf.confusion_matrix(labels, predictions)
    confusionmatrix = confusion.eval(session=tf.Session())
    return confusionmatrix

def calculateMCC(confusionMatrix):
    tp = confusionMatrix[0,0]
    tn = confusionMatrix[1,1]
    fp = confusionMatrix[0,1]
    fn = confusionMatrix[1,0]
    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return mcc

def plot_loss_history(histories, epochs, name):
    val_loss = np.zeros((epochs, 1))
    train_loss = np.zeros((epochs, 1))
    for history in histories:
        val_loss = val_loss + np.array(history.history['val_loss'])
        train_loss = train_loss + np.array(history.history['loss'])
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    tikz_save(name+".tex")
