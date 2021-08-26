import pandas as pd
import sys
mylib = '/home/fra/Project/pyProj/mlib'
sys.path.append(mylib)

data_folder = "/home/fra/DataMart/datacentre/opendata/UCI/"
train_filename = 'abalone.csv'

# ------------------------------------------------------------
# load data
# ------------------------------------------------------------
df = pd.read_csv(data_folder + train_filename, skipinitialspace=True)
# encode target
mapping = {'M':0, 'F':1, 'I':2}

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2)

import model as ml
import pipeline as pl

exclusions=[]
target_col = "SEX"

# ------------------------------------------------------------
# pipeline
# ------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
stdpl = pl.StandardPipeline(target_col, exclusions)
stdpl.make_pipeline(ntransformer=StandardScaler, ctransformer=OrdinalEncoder)

(X_train, y_train) = stdpl.fit_transform(df_train)
(X_test, y_test) = stdpl.transform(df_test)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
import xgboost as xgb

xgb_cls = xgb.XGBClassifier(use_label_encoder=False)
model = ml.MultiClassifier(xgb_cls)
model.fit(X_train, y_train)

xgb_eval = ml.MultiClassificationEvaluator(model)
results = xgb_eval.performance_summary(X_test, y_test)
print(results)

# {'Accuracy': 0.5119617224880383, 
# 'Consfusion Matrix': array([[103,  35, 119],
#        [ 31, 183,  49],
#        [115,  59, 142]]), 
# 'AUC': {0: 0.6783129372391685, 1: 0.8655930032714217, 2: 0.640153359298929}, 
# 'GINI': {0: 0.35662587447833705, 1: 0.7311860065428435, 2: 0.28030671859785805}}