
from Libraries import configuration
from Libraries import data_import as dataImport
from Libraries import data_preprocessing as dataPreproc
from Libraries import prepare_csv as csv
import pickle
from Libraries import log_setup as logSetup
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
import numpy as np
import pandas as pd

projectName = 'FindArchitecture'
callDataset = '1051'
configurations = [configuration.getConfig(projectName, callDataset)]

for config in configurations:
    # Setup Logger
    logSetup.configureLogfile(config.logPath, config.logName)
    logSetup.writeLogfileHeader(config)

    # Import verified Time Series Data
  #  X_ts, labels = dataImport.loadVerifiedBrakeData(config.eedPath, config.eecPath, config.datasetNumber)

  #  dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']

   # eecData = csv.eec_csv_to_eecData(config.eecPath, config.datasetNumber)
   # labels = dataPreproc.getTimeDistributedLabels(eecData, X_ts, flag=0)

    Data = pickle.load(open(config.picklePath + 'Data',"rb"))

    v = 1

    #X_ts, labels = dataPreproc.truncate_all(X_ts, labels, duration=100, part='center', discard=False)
    #labels_single = dataPreproc.reduceLabel(labels)
    #X = dataPreproc.shape_Data_to_LSTM_format(X_ts, dropChannels)
    #y = dataPreproc.shape_Labels_to_LSTM_format(labels)
    #k = dataPreproc.shape_Labels_to_LSTM_format(labels_single)
    #v = X.shape
    #m = Sequential()
    #m.add(LSTM(13, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    #m.add(Flatten())
    #m.add(Dense(1, activation='sigmoid'))
    #m.add(TimeDistributed(Dense(1, activation='sigmoid')))
    #m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'matthews_correlation'])
    #m.fit(X, k, batch_size=20, epochs=5)

