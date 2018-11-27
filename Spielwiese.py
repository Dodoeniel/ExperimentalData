
from Libraries import configuration
from Libraries import data_import as dataImport
from Libraries import data_preprocessing as dataPreproc
from Libraries import prepare_csv as csv
from Libraries import pickling
from Libraries import log_setup as logSetup
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

    dropChannels = ['trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
    X_ts = dataPreproc.dropDataChannels(X_ts, dropChannels)

    eecData = csv.eec_csv_to_eecData(config.eecPath, config.datasetNumber)
    labels = dataPreproc.getTimeDistributedLabels(eecData, X_ts, flag=0)

    discard = 1
    w_length = 100
    hop_size = 10

    X_sliced, labels_sliced = dataPreproc.sliceData(X_ts, labels, w_length, hop_size, discard)

    truncate = [1, 1, 1]
    X_trunc = pd.DataFrame()
    labels_trunc = pd.DataFrame()
    for stopId in X_ts['stopId'].unique():
        # get the one series that shall be truncated
        X_snippet = X_ts.loc[X_ts['stopId'] == stopId]
        label_snippet = labels.loc[labels['stopId'] == stopId]
        if len(X_snippet) >= w_length or (len(X_snippet) < w_length and discard == 0):

    k = 1
