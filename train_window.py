import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from Libraries import data_preprocessing as pp
from Libraries import data_evaluation as eval
from Libraries import model_evaluation as m_Eval
from sklearn.model_selection import train_test_split
from Libraries import model_setup

# list of names for Runs
RunNames = ['SmallWindowData']
# list of data set names
fileNames = ['SmallWindowData.p']
# list of models to run
models = ['singleLabel_1']
path = '/home/computations/ExperimentalData/'

history = list()
for i in range(len(RunNames)):
    RunName = RunNames[i]
    file = open(fileNames[i], 'rb')
    Data = pickle.load(file)
    dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
    X = pp.shape_Data_to_LSTM_format(Data[0], dropChannels)
    y = pp.reduceLabel(Data[1]).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    epochs = 100
    batch_size = 10

    m = Sequential()
    input_shape = (X.shape[1], X.shape[2])
    m = model_setup.modelDict[models[i]](input_shape)
    class_weight = {0: 1.,
                    1: 10.
		   }
    history.append(m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0, class_weight=class_weight))

    m_Eval.eval_all(history, epochs, RunName, m, path)


