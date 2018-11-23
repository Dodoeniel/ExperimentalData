
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

    # Smoothing of Time Series Data
    X_ts_smoothed = dataPreproc.smoothingEedData(X_ts)

    # Documentation
    pickling.writeDataToPickle(X_ts, config.picklePath + '/X_ts')
    pickling.writeDataToPickle(labels, config.picklePath + '/labels')

    logSetup.resetLogConfigurations()  # to get a new logfile for next configuration