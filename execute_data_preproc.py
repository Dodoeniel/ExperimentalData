from Libraries import configuration
from Libraries import data_import as dataImport
from Libraries import data_preprocessing as dataPreproc
from Libraries import prepare_csv as csv
import pickle

from Libraries import log_setup as logSetup



projectName = 'FindArchitecture'
callDataset = '1051'
config = configuration.getConfig(projectName, callDataset)

# Setup Logger
logSetup.configureLogfile(config.logPath, config.logName)
logSetup.writeLogfileHeader(config)

# Import verified Time Series Data with Nadines Libraries
X_ts, labels = dataImport.loadVerifiedBrakeData(config.eedPath, config.eecPath, config.datasetNumber)
X_ts = dataPreproc.smoothingEedData(X_ts)
eec = csv.eec_csv_to_eecData(config.eecPath, callDataset)


# data preproc with Daniels Libraries
labels = dataPreproc.getTimeDistributedLabels(eec, X_ts)

target_list = [(11, 1), (10, 1), (9, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1)]
part = ['center', 'center', 'center', 'center', 'center', 'center', 'center', 'center']

Data = dataPreproc.truncate_differentiated(X_ts, labels, part, target_list)


pickle.dump(Data, open(config.picklePath + '/Data', "wb"))

