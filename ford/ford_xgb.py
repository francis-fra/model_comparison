import xgboost as xgb
import numpy as np
import pandas as pd
import sys
mylib = '/home/fra/Project/pyProj/zqlib/zq/'
sys.path.append(mylib)
from dataframe import evaluator as ml

def get_dataframe():

    def readucr(filename):
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        # transform target
        f = lambda x: 0 if -x == -1 else 1
        vf = np.vectorize(f)
        y = vf(y)
        return x, y.astype(int)

    data_folder = '/home/fra/DataMart/datacentre/opendata/time_series/FordA/'
    train_filename = 'FordA_TRAIN.tsv'
    test_filename = 'FordA_TEST.tsv'

    x_train, y_train = readucr(data_folder + train_filename)
    x_test, y_test = readucr(data_folder + test_filename)

    return (x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train, x_test, y_test) = get_dataframe()

    model = ml.BinaryClassifier(xgb.XGBClassifier(use_label_encoder=False))
    xgb_eval = ml.BinaryClassificationEvaluator(model)
    xgb_eval.fit(x_train, y_train).validate(x_test)

    results = xgb_eval.performance_summary(y_test)
    print(results)
    # {'Accuracy': 0.7704545454545455, 'Consfusion Matrix': array([[475, 164],
    #    [139, 542]]), 'AUC': 0.8526239834175554, 'Gini': 0.7052479668351108}