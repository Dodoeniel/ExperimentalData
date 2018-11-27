
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

    discard = 0
    w_length = 100
    hop_size = 10
    for stopId in X_ts['stopId'].unique():
        # get the one series that shall be sliced
        X_snippet = X_ts.loc[X_ts['stopId'] == stopId]
        label_snippet = labels.loc[labels['stopId'] == stopId]
        # pre define output
        X_sliced = pd.DataFrame()
        labels_sliced = pd.DataFrame()
        if len(X_snippet) >= w_length or (len(X_snippet) < w_length and discard == 0):
            hop = 0
            for hop in range((len(X_snippet)-w_length)//hop_size):

                # create Ids for sliced TS:  1.051_0 .... 1.051_0 1.051_1 .... 1.051_1
                sliceId = []
                for i in range(w_length):
                    sliceId.append(stopId + '_' + str(hop))

                # slice labels
                curr_label = label_snippet[hop * hop_size:hop * hop_size + w_length]
                curr_label.index = range(w_length)
                curr_label['sliceId'] = pd.Series(sliceId, index=curr_label.index)
                labels_sliced = pd.concat([labels_sliced, curr_label])

                # slice data
                curr_X = X_snippet[hop * hop_size:hop * hop_size + w_length]
                curr_X.index = range(w_length)
                curr_X['sliceId'] = pd.Series(sliceId, index=curr_X.index)
                X_sliced = pd.concat([X_sliced, curr_X])

            if (len(X_snippet)-w_length) % hop_size > 0 & discard == 0: # remainder of windowed function

                if hop != 0:
                    hop += 1
                    curr_label = label_snippet[hop * hop_size + w_length:]
                    curr_X = X_snippet[hop * hop_size + w_length:]
                else:
                    curr_label = label_snippet
                    curr_X = X_snippet

                sliceId = stopId + '_' + str(hop)

                curr_label.index = range(len(curr_label))
                zero_padding = pd.DataFrame(np.zeros(((w_length - len(curr_label)), curr_label.shape[1])),
                                            index=range(len(curr_label), w_length),
                                            columns=curr_label.columns.values.tolist())
                curr_label = pd.concat([curr_label, zero_padding])
                curr_label['sliceId'] = pd.Series(sliceId, index=curr_label.index)
                labels_sliced = pd.concat([labels_sliced, curr_label])

                # remaining data
                curr_X.index = range(len(curr_X))

                zero_padding = pd.DataFrame(np.zeros(((w_length - len(curr_X)), curr_X.shape[1])),
                                            index=range(len(curr_X), w_length),
                                            columns=curr_X.columns.values.tolist())
                curr_X = pd.concat([curr_X, zero_padding])
                curr_X['sliceId'] = pd.Series(sliceId, index=curr_X.index)
                X_sliced = pd.concat([X_sliced, curr_X])
        elif discard == 0: ## TODO obsolete
            curr_label = label_snippet
            curr_label.index = range(len(curr_label))
            zero_padding = pd.DataFrame(np.zeros(((w_length-len(curr_label)), curr_label.shape[1])),
                                        index=range(len(curr_label), w_length), columns=curr_label.columns.values.tolist())
            curr_label = pd.concat([curr_label, zero_padding])
            curr_label['sliceId'] = pd.Series(sliceId, index=curr_label.index)
            labels_sliced = pd.concat([labels_sliced, curr_label])
            v = 1

        x = 1



    print(X_ts)
