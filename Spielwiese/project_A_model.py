"""Train Model for project A"""

from brakesqueal_library import pickling
from brakesqueal_library import configuration
from brakesqueal_library import model_setup as modelSetup
from brakesqueal_library import model_evaluation as modelEval
from brakesqueal_library import log_setup as logSetup
from brakesqueal_library import data_preprocessing as dataPreproc

import sys
import logging

if __name__ == '__main__':

    # Get Configuration
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

    for config in configurations:

        # Setup Logger
        logSetup.configureLogfile(config.logPath, config.logName)
        logSetup.writeLogfileHeader(config)

        # Get Data
        X_train = pickling.readDataFromPickle(config, 'X_feat_relevant_norm_train')
        X_test  = pickling.readDataFromPickle(config, 'X_feat_relevant_norm_test')
        X_val   = pickling.readDataFromPickle(config, 'X_feat_relevant_norm_val')

        labels_train = pickling.readDataFromPickle(config, 'labels_train')
        labels_test  = pickling.readDataFromPickle(config, 'labels_test')
        labels_val   = pickling.readDataFromPickle(config, 'labels_val')

        # Ensure Pickling did not change order in X and labels and thus assignments
        pickling.assertLabelAssignmentIsUnchanged(X_train, labels_train)
        pickling.assertLabelAssignmentIsUnchanged(X_test,  labels_test)
        pickling.assertLabelAssignmentIsUnchanged(X_val,   labels_val)

        #...

        logSetup.resetLogConfigurations()