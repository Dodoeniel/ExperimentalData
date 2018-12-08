"""Library for Evaluation of Input Data"""
import matplotlib.pyplot as plt
import math
def count_predictions(y_true, y_pred):
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for k in range(y_pred.shape[0]):
        y_true_curr = y_true[k]
        y_pred_curr = y_pred[k]
        for l in range(len(y_true[k])):
            if (y_true[k][l] == y_pred[k][l]) & (y_pred[k][l] == 0):
                TN += 1
            elif (y_true[k][l] == y_pred[k][l]) & (y_pred[k][l] == 1):
                TP += 1
            elif y_true[k][l] == 1:
                FN += 1
            elif y_pred[k][l] == 1:
                FP += 1
    return FP, FN, TP, TN

def get_overall_results(test_data, model):
    FP = 1
    FN = 1
    TP = 1
    TN = 1
    for currData in test_data:
        X_test = currData[0]
        y = currData[1]
        FP_loop, FN_loop, TP_loop, TN_loop = count_predictions(y, model.predict_classes(X_test))
        FP += FP_loop
        FN += FN_loop
        TP += TP_loop
        TN += TN_loop
    print('\nMCC: ' + str(get_MCC(FP, FN, TP, TN)))
    print('\n' + str(TP) + '  ' + str(FN))
    print('\n' + str(FP) + '  ' + str(TN))
    return FP, FN, TP, TN

def get_MCC(FP, FN, TP, TN):
    try:
        return (TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    except ZeroDivisionError:
        return 0