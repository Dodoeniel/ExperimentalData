
from Libraries import configuration
from Libraries import data_import as dataImport
from Libraries import data_preprocessing as dataPreproc
from Libraries import prepare_csv as csv
from Libraries import pickling
from Libraries import log_setup as logSetup
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
import numpy as np
import pandas as pd

projectName = 'Spielwiese'
callDataset = '1051'
configurations = [configuration.getConfig(projectName, callDataset)]

for config in configurations:
    # Setup Logger
    logSetup.configureLogfile(config.logPath, config.logName)
    logSetup.writeLogfileHeader(config)

    # Import verified Time Series Data
    X_ts, labels = dataImport.loadVerifiedBrakeData(config.eedPath, config.eecPath, config.datasetNumber)

    dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']

    eecData = csv.eec_csv_to_eecData(config.eecPath, config.datasetNumber)
    labels = dataPreproc.getTimeDistributedLabels(eecData, X_ts, flag=0)


    w_length = 100
    hop_size = 10



    target = (11, 2)

    duration = target[0]
    part = 'center'
    discard = True
    X_truncated = pd.DataFrame()
    labels_truncated = pd.DataFrame()
    # iterate over all time sersies
    for stopId in X_ts['stopId'].unique():
        X_to_truncate = X_ts.loc[X_ts['stopId'] == stopId]
        label_to_truncate = labels.loc[labels['stopId'] == stopId]
        check = X_to_truncate.get_value(len(X_to_truncate) -1, 'time') >= target[0]
        check2 = X_to_truncate.get_value(len(X_to_truncate) -1, 'time') <= (target[0] + target[1])
        if check & check2:
            # call function that truncates a single time series and distributed label
            X_single, label_single = dataPreproc.truncate_single(X_to_truncate, label_to_truncate, target[0], part, discard)
            # adding truncated time series and label to data frame, if not nan (to short for window)
            if not np.isnan(X_single):
                X_truncated = pd.concat([X_truncated, X_single])
                labels_truncated = pd.concat([labels_truncated, label_single])
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

