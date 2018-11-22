"""Prepare Data for project A

For Use on Cluster: Pass Name of Dataset , 'e.g.: 1085' as Parameter
"""
from brakesqueal_library import configuration
from brakesqueal_library import data_import as dataImport
from brakesqueal_library import data_preprocessing as dataPreproc
from brakesqueal_library import pickling
from brakesqueal_library import log_setup as logSetup

import sys

if __name__ == '__main__':

    # get configurations, i.e. datasets with paths and names, etc.
    projectName = 'Project_A'

    if len(sys.argv)>1:
        callDataset = sys.argv[1]
        configurations = [configuration.getConfig(projectName, callDataset)]
    else:
        configurations = [
            configuration.getConfig_1085(projectName),
            configuration.getConfig_1093(projectName),
            configuration.getConfig_1114(projectName),
            configuration.getConfig_1131(projectName),
        ]

    # get data for each configuration
    for config in configurations:

        # Setup Logger
        logSetup.configureLogfile(config.logPath, config.logName)
        logSetup.writeLogfileHeader(config)

        # Import verified Time Series Data
        X_ts, labels = dataImport.loadVerifiedBrakeData(config.eedPath, config.eecPath, config.datasetNumber)

        # Smoothing of Time Series Data
        X_ts_smoothed = dataPreproc.smoothingEedData(X_ts)

        # Documentation
        pickling.writeDataToPickle(X_ts,    config.picklePath + '/X_ts')
        pickling.writeDataToPickle(labels,  config.picklePath + '/labels')

        logSetup.resetLogConfigurations() #to get a new logfile for next configuration