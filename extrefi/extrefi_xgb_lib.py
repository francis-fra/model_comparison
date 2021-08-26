import pandas as pd
import os, sys
mlibpath = r"/home/fra/Project/pyProj/mlib"
sys.path.append(mlibpath)

import utility as ut
import explore as ex 
import model as ml
import pipeline as pl

# ------------------------------------------------------------
# load data
# ------------------------------------------------------------
data_folder = "/home/fra/DataMart/datacentre/westpac/"
train_filename = "W_EXTREF_R_V_MDS_train.csv"
data_file = data_folder + train_filename
# data_file

df = pd.read_csv(data_file)

test_filename = "W_EXTREF_R_V_MDS_test.csv"
data_file = data_folder + test_filename
df_test = pd.read_csv(data_file)

# xgb_model = xgb.XGBClassifier(n_jobs=num_workers, use_label_encoder=False)
# xgb_model.fit(X_train, y_train)

# ------------------------------------------------------------
# define pipeline
# ------------------------------------------------------------
exclusions=['DATA_DT', 'PROCESS_DTTM', 'GCIS_KEY', 'CUSTOMER_ID', 'PERIOD_ID']
target_col ="TARGET_F"

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
stdpl = pl.StandardPipeline(target_col, exclusions)

# train pipeline
stdpl.process(df, ntransformer=StandardScaler, ctransformer=OrdinalEncoder)
# FIXME
(X, y) = stdpl.out()
# stdpl.target_col
# stdpl.mapping
# stdpl.features
# len(stdpl.features)
# X.shape

# test pipeline
stdpl.process(df_test, ntransformer=StandardScaler, ctransformer=OrdinalEncoder)
(X_test, y_test) = stdpl.out()
