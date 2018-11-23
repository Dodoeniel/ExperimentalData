
from Libraries import configuration
from Libraries import data_import as dataImport
from Libraries import data_preprocessing as dataPreproc
from Libraries import pickling
from Libraries import log_setup as logSetup

projectName = 'Spielwiese'
callDataset = '1051'
configurations = [configuration.getConfig(projectName, callDataset)]

for config in configurations:
    # Setup Logger
    logSetup.configureLogfile(config.logPath, config.logName)
    logSetup.writeLogfileHeader(config)

    # Import verified Time Series Data
    X_ts, labels = dataImport.loadVerifiedBrakeData(config.eedPath, config.eecPath, config.datasetNumber)

    dropChannels = ['stopId', 'time', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
    X_ts = dataPreproc.dropDataChannels(X_ts, dropChannels)
    print(X_ts)
