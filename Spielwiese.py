
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

    X_trunc, label_trunc = dataPreproc.truncate_all(X_ts, labels, 0.1, 'center')

    k = 1
