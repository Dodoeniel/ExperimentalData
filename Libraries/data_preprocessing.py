"""library for data preprocessing steps for time series

@author: Nadine

# features erzeugen
# smoothen ( time series bearbeiten)
# normalizieren ( time series oder feauters)
# outlier detection...
# upsampling
"""

import random
import collections
import pandas as pd
import numpy as np

#import tsfresh
#import tsfresh.utilities.dataframe_functions as tsfreshUtil

import logging


# tsFresh based feature generation
def generateFeaturesFromTimeSeries(X_ts):
    """
    Parameters:
        X_ts: Flat DataFrame
            Contains time series Data from which features should be extracted
            Example:
            (index)  stopId  time quantity1 quantity2
                     --------------------------------
              0      1.1501   1     ...       ...
              1      1.1501   2     ...       ..
              2      1.1501   3     ...       ...
              3      2.1501   1     ...       ...
              4      2.1501   2     ...       ...

    Returns:
        X_feat: DataFrame
            Contains extracted Features and their values

            (index)    f1(quantity1)  f2(quantity1)  f1(quantity2)  f3(quantity2)
            (stopId)
                       ----------------------------------------------------------
            1.1501       ...            ...            ...            ...
            2.1501       ...            ...            ...            ...
            3.1501       ...            ...            ...            ...
    """

    X_feat = tsfresh.extract_features(X_ts, column_id="stopId", column_sort="time", n_jobs=12)

    # Replac NaN- Values with 0
    tsfreshUtil.impute(X_feat)
    # Replac index 'id' with 'stopId'
    X_feat.index.name = "stopId"

    return X_feat

def selectRelevantFeatures(X_feat, labels):
    """
    Parameters:
        X_feat: DataFrame
            Contains extracted Features and their values

            (index)    f1(quantity1)  f2(quantity1)  f1(quantity2)  f3(quantity2)
                       ----------------------------------------------------------
            1.1501       ...            ...            ...            ...
            2.1501       ...            ...            ...            ...
            3.1501       ...            ...            ...            ...

        labels: Series
            Contains labels for time series Data
                0: No Squeal, 1: Squeal

            (index)  label
                     -----
             1.1501    0
             2.1501    0
             3.1501    1
             4.1501    1
             5.1501    0

    Returns:
        X_feat_relevant: DataFrame
            Contains relevant features and their values

            (index)    f1(quantity1)  f1(quantity2)  f3(quantity2)
                       -------------------------------------------
            1.1501       ...            ...            ...
            2.1501       ...            ...            ...
            3.1501       ...            ...            ...
    """
    X_feat_relevant = tsfresh.select_features(X_feat, labels)

    summarizeFeatureSelection(X_feat, X_feat_relevant)

    numberOfRelevantFeatures = X_feat_relevant.shape[1]
    if numberOfRelevantFeatures == 0:
        logging.error("Did not find relevant features.")
        #raise ValueError("Did not find relevant features.")

    return X_feat_relevant

def getFeatureNames(X_feat):
    """
    Parameters:
        X_feat: DataFrame
            Contains extracted Features and their values

            (index)    f1(quantity1)  f2(quantity1)  f1(quantity2)  f3(quantity2)
                       ----------------------------------------------------------
            1.1501       ...            ...            ...            ...
            2.1501       ...            ...            ...            ...
            3.1501       ...            ...            ...            ...

    Returns:
        featureNames: List
            Contains names of all Features stored in X_feat
    """

    featureNames = X_feat.columns.tolist()
    return featureNames

def getSubsetOfFeatures(X_feat, featuresToInclude):

    X_feat_subset = X_feat[featuresToInclude]

    return  X_feat_subset


# Splitting of Data
def splitData(X, labels, val_split = 0.2, test_split = 0.2):

    assert(val_split + test_split < 1)

    numOfTotalExmpls   = X.shape[0]
    numOfExmplsInVal   = round(val_split  * numOfTotalExmpls)
    numOfExmplsInTest  = round(test_split * numOfTotalExmpls)
    numOfExmplsInTrain = numOfTotalExmpls - numOfExmplsInVal - numOfExmplsInTest

    assert(numOfExmplsInTrain > 0)

    X_shuffled, y_shuffled = shuffleData(X, labels)

    X_train = X_shuffled[:numOfExmplsInTrain]
    y_train = y_shuffled[:numOfExmplsInTrain]

    X_test  = X_shuffled[numOfExmplsInTrain:(numOfExmplsInTrain + numOfExmplsInTest)]
    y_test  = y_shuffled[numOfExmplsInTrain:(numOfExmplsInTrain + numOfExmplsInTest)]

    X_val   = X_shuffled[(numOfExmplsInTrain + numOfExmplsInTest):]
    y_val   = y_shuffled[(numOfExmplsInTrain + numOfExmplsInTest):]

    # Create Classes: collections.namedtuple(classname, constructor)
    TrainData = collections.namedtuple('trainData', ['X', 'labels'])
    TestData  = collections.namedtuple('testData',  ['X', 'labels'])
    ValData   = collections.namedtuple('valData',   ['X', 'labels'])

    # Instanciate Classes
    trainData = TrainData(X_train, y_train)
    testData  = TestData(X_test, y_test)
    valData   = ValData(X_val, y_val)

    summarizeDataSplitting(trainData, testData, valData)

    return trainData, testData, valData

def shuffleData(X, labels):
    """X: DataFrame with unique indices (bzw wird zeilenweise gesplittet, nicht fuer flatDF geeignet)"""

    assert(sorted(X.index.tolist()) == sorted(labels.index.tolist()))

    indices = X.index.tolist()
    random.shuffle(indices)
    X_shuffled = X.loc[indices]
    labels_shuffled = labels.loc[indices]

    return X_shuffled, labels_shuffled


# Balancing of Data
def getBalancedDataByDropping(X, labels, targetPercent, reference):

    X.sort_index(inplace=True)
    labels.sort_index(inplace=True)

    assert (X.index.tolist() == labels.index.tolist())
    assert (targetPercent <=100)
    # annahme: Squeal ist unterbesetzt

    X_label_0 = X[labels == 0]
    y_label_0 = labels[labels == 0]

    X_label_1 = X[labels == 1]
    y_label_1 = labels[labels == 1]

    (X_label_0_shuffled, y_label_0_shuffled) = shuffleData(X_label_0, y_label_0)
    (X_label_1_shuffled, y_label_1_shuffled) = shuffleData(X_label_1, y_label_1)

    numOfSquealExamples    = len(X_label_1)
    numOfNoSquealsExamples = len(X_label_0)

    if(reference == 'Squeal'):

        numOfNoSquealsExamples = round(numOfSquealExamples * (100-targetPercent) / targetPercent)
        X_0_new = X_label_0_shuffled[:numOfNoSquealsExamples]
        y_0_new = y_label_0_shuffled[:numOfNoSquealsExamples]

        X_balanced = pd.concat([X_label_1, X_0_new])
        y_balanced = pd.concat([y_label_1, y_0_new])

    elif(reference == 'NoSqueal'):

        numOfSquealExamples = round(numOfNoSquealsExamples * (100-targetPercent) / targetPercent)
        X_1_new = X_label_1_shuffled[:numOfSquealExamples]
        y_1_new = X_label_1_shuffled[:numOfSquealExamples]

        X_balanced = pd.concat([X_label_0, X_1_new])
        y_balanced = pd.concat([y_label_0, y_1_new])

    else:
        X_balanced = None
        y_balanced = None

    summarizeBalancing(labels, y_balanced)
    return X_balanced, y_balanced


# Data Normalization
def normalizeFeatureDatasetsScaledByStd(X_feat_train, X_feat_test, X_feat_val):
        meanOfTrain = X_feat_train.mean()
        stdOfTrain  = X_feat_train.std()

        X_feat_train_normalized = normalizeFeaturesScaledByStd(X_feat_train, meanOfTrain, stdOfTrain)
        X_feat_test_normalized  = normalizeFeaturesScaledByStd(X_feat_test, meanOfTrain, stdOfTrain)
        X_feat_val_normalized   = normalizeFeaturesScaledByStd(X_feat_val, meanOfTrain, stdOfTrain)

        return X_feat_train_normalized, X_feat_test_normalized, X_feat_val_normalized

def normalizeFeaturesScaledByStd(X_feat, mean, std):
    """ Scaling and mean normalization of features using std deviation

    Parameters:
        X_feat: DataFrame
            Contains original values.
            One row per example and one column per feature

            (index)   f1   f2   f3
                    -----------------
            1.1051   ...  ...  ...
            2.1051   ...  ...  ...
            3.1051   ...  ...  ...

        mean: Series
            Contains mean for each feature

            (index)  mean
                    ------
               f1     ..
               f2     ..
               f3     ..

        std: Series
            Contains standard deviation for each feature

            (index)  std
                    ------
               f1     ..
               f2     ..
               f3     ..

    Returns:
        X_feat_normalized: DataFrame
            Contains normalized values.
            One row per example and one column per feature.

            (index)   f1   f2   f3
                    -----------------
            1.1051   ...  ...  ...
            2.1051   ...  ...  ...
            3.1051   ...  ...  ...
    """

    std[std == 0] = 1  # avoid to divide by zero
    X_feat_normalized = (X_feat - mean)/std

    return X_feat_normalized

def normalizeFeatureDatasetsScaledByRange(X_feat_train, X_feat_test, X_feat_val):
    maxOfTrain  = X_feat_train.max()
    minOfTrain  = X_feat_train.min()
    meanOfTrain = X_feat_train.mean()

    X_feat_train_normalized = normalizeFeaturesScaledByRange(X_feat_train, meanOfTrain, maxOfTrain, minOfTrain)
    X_feat_test_normalized  = normalizeFeaturesScaledByRange(X_feat_test,  meanOfTrain, maxOfTrain, minOfTrain)
    X_feat_val_normalized   = normalizeFeaturesScaledByRange(X_feat_val,   meanOfTrain, maxOfTrain, minOfTrain)

    return X_feat_train_normalized, X_feat_test_normalized, X_feat_val_normalized

def normalizeFeaturesScaledByRange(X_feat, mean, max, min):
    """ Scaling and mean normalization of features using range

    Parameters:
        X_feat: DataFrame
            Contains original values.
            One row per example and one column per feature

            (index)   f1   f2   f3
                    -----------------
            1.1051   ...  ...  ...
            2.1051   ...  ...  ...
            3.1051   ...  ...  ...

        mean: Series
            Contains mean for each feature

            (index)  mean
                    ------
               f1     ..
               f2     ..
               f3     ..

        max: Series
            Contains max value for each feature

            (index)  max
                    ------
               f1     ..
               f2     ..
               f3     ..

        min: Series
            Contains min value for each feature

            (index)  min
                    ------
               f1     ..
               f2     ..
               f3     ..

    Returns:
        X_feat_normalized: DataFrame
            Contains normalized values.
            One row per example and one column per feature.

            (index)   f1   f2   f3
                    -----------------
            1.1051   ...  ...  ...
            2.1051   ...  ...  ...
            3.1051   ...  ...  ...
    """
    valueRange = (max-min)
    valueRange[valueRange == 0] = 1  # avoid to divide by zero
    X_feat_normalized = (X_feat - mean)/valueRange

    return X_feat_normalized


# Smoothing of Data
def smoothingEedData(X_ts):

    smoothedEedData = []
    for stopId, eedDataOfStopId in X_ts.groupby('stopId', sort=False):
        smoothedEedDataOfStopId = smoothingEedDataForStopId(eedDataOfStopId)
        smoothedEedData.append(smoothedEedDataOfStopId)

    X_ts_smoothed = pd.concat(smoothedEedData)

    return X_ts_smoothed

def smoothingEedDataForStopId(eedDataOfStopId):
    """smoothes Data from one eed file"""

    smoothedEedDataForStopId = []
    excludedColumns = ['stopId', 'time']

    for columnname in eedDataOfStopId.columns:

        column = eedDataOfStopId[columnname]

        if columnname not in excludedColumns:
            windowSize     = getSmoothingWindowSizeForColumn(columnname)
            columnSmoothed = smoothingSeries(column, windowSize)
            smoothedEedDataForStopId.append(columnSmoothed)
        else:
            smoothedEedDataForStopId.append(column)

    # Create Data Frame from Smoothed Columns
    entriesOfDict   = [(series.name, series) for series in smoothedEedDataForStopId] # [(key1, val1), (key2, val2),..]
    eedDataAsDict   = collections.OrderedDict(entriesOfDict)
    smoothedEedData = pd.DataFrame.from_dict(eedDataAsDict)

    return smoothedEedData

def smoothingSeries(column, rollingWindowsize):
    # center = true -> sonst versatz
    # min period = 1 -> enden keine NaN values
    column_smooth = column.rolling(rollingWindowsize, min_periods=1, center=True).mean()
    return column_smooth

def getSmoothingWindowSizeForColumn(columnname):
    defaultWindowSize = 3
    windowSizesForColumn = {
        'p1': 3,
        'n1': 3
    }
    if columnname in windowSizesForColumn:
        windowSize = windowSizesForColumn[columnname]
    else:
        windowSize = defaultWindowSize

    return  windowSize


# Daniels functions
def dropDataChannels(X_ts, channel_names):
    """
    Drops specific columns from the data set
    :param X_ts: all operational parameter data Nadines Format X_ts
    @author: Daniel
    :param channel_names: vector of not wanted channels by name
    :return: X_ts with omitted columns
    """
    return(X_ts.drop(channel_names, axis=1))


def  getTimeDistributedLabels(eec_data,X_ts, flag=0):
    """
    extends the labels towards time distributedLabels,
    flag for type of labeling: 0,1
    :return:
    """
    FLAG1 = 0  # Flag for binary classification, y/n
    FLAG2 = 1  # Flag for classification with uprising downgoing

    labels = X_ts['StopId']
    labels['label'] = pd.Series(np.zeros(len(labels['StopId']), 1), index=labels.index)

    for i in eec_data.index:
        stopId = eec_data.get_value(i, 'StopId')    # fastes way to iterate
        if not eec_data(['d_1']).isnull:
            nrSqueals = 0
            while True:
                # iterate of all possible squealing
                nrSqueals += 1
                if not eec_data(['d_'+str(nrSqueals)]).isnull:
                    start = eec_data['time_' + str(nrSqueals) + '&start']
                    stop = eec_data['time_' + str(nrSqueals) + '&end']
                else:
                    # break if no squealing occurs
                    break
                # find indexes of time steps with noise in it
                # possible alternative with between() or query()
                indexes = X_ts.index[(X_ts['StopId'] == stopId) & (X_ts['time'] >= start) & (X_ts['time'] <= stop)]
                labels[indexes] = np.ones(len(indexes), 1)
                #for l in len(indexes):# TODO schöner machen
                #    labels[indexes(l)] = 1





# Log Functions
def summarizeFeatureSelection(X_feat, X_feat_relevant):

    # todo: tsfresh output in separate file?
    numExtractedFeatures = X_feat.shape[1]
    numSelectedFeatures = X_feat_relevant.shape[1]
    noOfExamplesUsedForSelection = X_feat_relevant.shape[0]
    irrelevantFeatures = numExtractedFeatures - numSelectedFeatures
    title = "Summary of Feature Generation: \n"

    summarySelection = "  Selected Features: " + str(numSelectedFeatures) + "/" + str(numExtractedFeatures) +\
                       " (" + str(irrelevantFeatures) + " irrelevant)" + "\n" + \
                       "  Examples used for Selection: " + str(noOfExamplesUsedForSelection)

    listOfFeatures = sorted(getFeatureNames(X_feat_relevant))
    selectedFeatures = '\n'.join('{}: {}'.format(*k) for k in enumerate(listOfFeatures))

    logmsg = "\n".join([title, summarySelection, selectedFeatures, "\n"])

    logging.info(logmsg)

def summarizeDataSplitting(trainData, testData, valData):

    numOfTrainData= len(trainData.X)
    numOfTestData = len(testData.X)
    numOfValData  = len(valData.X)
    numOfTotalExamples = numOfTrainData + numOfTestData + numOfValData

    percentTraining = round(numOfTrainData/numOfTotalExamples * 100)
    percentTest     = round(numOfTestData/numOfTotalExamples * 100)
    percentVal      = round(numOfValData/numOfTotalExamples * 100)

    numOfLabel1InTrainData = len(trainData.labels[trainData.labels == 1])
    numOfLabel1InTestData  = len(testData.labels[testData.labels == 1])
    numOfLabel1InValData   = len(valData.labels[valData.labels == 1])

    percentOfLabel1InTrain = round(numOfLabel1InTrainData/numOfTrainData * 100)
    percentOfLabel1InTest  = round(numOfLabel1InTestData/numOfTestData * 100)
    percentOfLabel1InVal   = round(numOfLabel1InValData/numOfValData * 100)

    numOfLabel0InTrainData = numOfTrainData - numOfLabel1InTrainData
    numOfLabel0InTestData  = numOfTestData - numOfLabel1InTestData
    numOfLabel0InValData   = numOfValData - numOfLabel1InValData

    title = "Summary of Data Splitting: \n"

    relations = "  Percentages for Splitting: " + "\n" + \
                "    Training: "   + str(percentTraining) +  "% (= " + str(numOfTrainData) + " Examples)" + "\n" + \
                "    Test: "       + str(percentTest)     +  "% (= " + str(numOfTestData)  + " Examples)" + "\n" + \
                "    Validation: " + str(percentVal)      +  "% (= " + str(numOfValData)   + " Examples)" + "\n"

    balance = "  Balance after Splitting: " + "\n" + \
              '    Squeals in Trainingset: '    +  str(numOfLabel1InTrainData) + "/" +  str(numOfTrainData) +\
                                              "(= " + str(percentOfLabel1InTrain) + " %) \n" +\
              '    Squeals in Testset: '        + str(numOfLabel1InTestData)  + "/" + str(numOfTestData) +\
                                              "(= " + str(percentOfLabel1InTest)  + " %) \n" + \
              '    Squeals in Validationset: '  + str(numOfLabel1InValData)   + "/" + str(numOfValData) +\
                                              "(= " + str(percentOfLabel1InVal)   + " %) \n"

    logmsg = "\n".join([title, relations, balance])

    logging.info(logmsg)

def summarizeBalancing(labels, labels_balanced):
    numOfNoSquealsOriginal     = len(labels[labels == 0])
    numOfSquealsOriginal       = len(labels[labels == 1])
    numOfTotalExamplesOriginal = len(labels)

    numOfNoSquealsBalanced     = len(labels_balanced[labels_balanced==0])
    numOfSquealsBalanced       = len(labels_balanced[labels_balanced==1])
    numOfTotalExamplesBalanced = len(labels_balanced)

    percentOfSquealsOriginal   = round(numOfSquealsOriginal/numOfTotalExamplesOriginal *100)
    percentOfSquealsBalanced   = round(numOfSquealsBalanced/numOfTotalExamplesBalanced * 100)
    percentOfNoSquealsBalanced = 100-percentOfSquealsBalanced
    percentOfNoSquealsOriginal = 100-percentOfSquealsOriginal

    numOfRemovedExamples = numOfTotalExamplesOriginal - numOfTotalExamplesBalanced

    if numOfTotalExamplesOriginal == numOfTotalExamplesBalanced:
        title = 'Keep Original Balance' + "\n"

    else:
        title = 'Change Balance of Data' + "\n"

    msg = title + "\n" + \
         "  Balance before Splitting:" + "\n" +\
         '    Squeals: '   + str(numOfSquealsBalanced)   + '/' + str(numOfTotalExamplesBalanced) +\
         " (" + str(percentOfSquealsBalanced) + '%)' + "\n" +\
         '    No Squeals: ' + str(numOfNoSquealsBalanced) + '/' + str(numOfTotalExamplesBalanced)  +\
         " (" + str(percentOfNoSquealsBalanced) + '%)' + "\n" +\
         "    Examples left: " + str(numOfTotalExamplesBalanced) +\
         ' (' + str(numOfRemovedExamples)+ "/" + str(numOfTotalExamplesOriginal) + " removed)" +"\n"

    logging.info(msg)
