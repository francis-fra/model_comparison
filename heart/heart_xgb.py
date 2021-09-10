import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

from sklearn.preprocessing import OneHotEncoder
import wpy as hp
import xgboost as xgb

def load_data():
    file_location = '/home/fra/DataMart/datacentre/opendata/UCI/heart.csv'
    df = pd.read_csv(file_location)
    return df

def evaluate_model(model, X, y):
    probas = model.predict(X)
    score = roc_auc_score(y, probas)
    pred = [1 if item > 0.5 else 0 for item in probas]
    acc = accuracy_score(y, pred)
    print(f"auc = {score:.4}, acc={acc:.4}")

def encode_data(df, datatype_dict):
    categorical_columns = []
    numerical_columns = []
    for name, datatype in datatype_dict.items():
        if datatype in ["integer", "categorical"]:
            categorical_columns.append(name)
        else:
            numerical_columns.append(name)

    X = df[categorical_columns].values
    Xnum = df[numerical_columns].values
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(X)
    return enc, categorical_columns, numerical_columns

def get_datatype_dict():
    integer_columns = ["sex", "cp", "fbs", "restecg", "exang", "ca"]
    string_columns = ["thal"]
    numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]
    datatype_dict = {}
    for col in integer_columns:
        datatype_dict[col] = "integer"
    for col in string_columns:
        datatype_dict[col] = "categorical"
    for col in numeric_columns:
        datatype_dict[col] = "numerical"
    return datatype_dict

def test_model_01():
    df = load_data()
    target = hp.get_single_column(df, 'target')
    pipeline = Pipeline([
                        ('Var Dropper', hp.VarDropper(excl=['target'])),
                        ('Imputer', hp.Imputer()),
                        ('Encoder', hp.Encoder())])
    df = pipeline.transform(df)
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

def test_model_02():
    df = load_data()
    target = hp.get_single_column(df, 'target')
    pipeline = Pipeline([
                        ('Var Dropper', hp.VarDropper(excl=['target'])),
                        ('Imputer', hp.Imputer()),
                        ('Encoder', hp.Encoder())])
    df = pipeline.transform(df)
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.3)
    model = xgb.XGBClassifier()

    # feature transformation
    datatype_dict = get_datatype_dict()
    enc, categorical_columns, numerical_columns = encode_data(X_train, datatype_dict)
    X_train_onehot = enc.transform(X_train[categorical_columns])
    X_test_onehot = enc.transform(X_test[categorical_columns])
    X_train_numeric = X_train[numerical_columns].values
    X_test_numeric = X_test[numerical_columns].values
    X_train_all = np.column_stack((X_train_onehot, X_train_numeric))
    X_test_all = np.column_stack((X_test_onehot, X_test_numeric))

    model.fit(X_train_all, y_train)
    evaluate_model(model, X_test_all, y_test)


if __name__ == '__main__':
    test_model_01()
    # auc = 0.6545, acc=0.7692
    test_model_02()
    # auc = 0.7333, acc=0.8242
