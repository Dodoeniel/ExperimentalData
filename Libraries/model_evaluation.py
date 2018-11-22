"""Contains Tools for Evaluating the Training Procedue and Helper Functions for Debugging and Documentation purpose"""
import matplotlib.pyplot as plt
import tensorflow as tf
import math

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
