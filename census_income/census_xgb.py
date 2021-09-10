import math
import wpy as hp
import xgboost as xgb

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.pipeline import Pipeline

from functools import partial

def load_data():
    CSV_HEADER = [
        "age",
        "class_of_worker",
        "detailed_industry_recode",
        "detailed_occupation_recode",
        "education",
        "wage_per_hour",
        "enroll_in_edu_inst_last_wk",
        "marital_stat",
        "major_industry_code",
        "major_occupation_code",
        "race",
        "hispanic_origin",
        "sex",
        "member_of_a_labor_union",
        "reason_for_unemployment",
        "full_or_part_time_employment_stat",
        "capital_gains",
        "capital_losses",
        "dividends_from_stocks",
        "tax_filer_stat",
        "region_of_previous_residence",
        "state_of_previous_residence",
        "detailed_household_and_family_stat",
        "detailed_household_summary_in_household",
        "instance_weight",
        "migration_code-change_in_msa",
        "migration_code-change_in_reg",
        "migration_code-move_within_reg",
        "live_in_this_house_1_year_ago",
        "migration_prev_res_in_sunbelt",
        "num_persons_worked_for_employer",
        "family_members_under_18",
        "country_of_birth_father",
        "country_of_birth_mother",
        "country_of_birth_self",
        "citizenship",
        "own_business_or_self_employed",
        "fill_inc_questionnaire_for_veteran's_admin",
        "veterans_benefits",
        "weeks_worked_in_year",
        "year",
        "income_level",
    ]
    train_file = '/home/fra/DataMart/datacentre/opendata/UCI/census/census_income_train.csv'
    test_file = '/home/fra/DataMart/datacentre/opendata/UCI/census/census_income_test.csv'

    df = pd.read_csv(train_file)
    testdf = pd.read_csv(test_file)

    df.columns = CSV_HEADER
    testdf.columns = CSV_HEADER

    # split validation and training
    random_selection = np.random.rand(len(df.index)) <= 0.85
    traindf = df[random_selection]
    validdf = df[~random_selection]

    return traindf, validdf, testdf

# target transfrom
def data_transform(df):
    df["income_level"] = df["income_level"].apply(
        lambda x: 0 if x == " - 50000." else 1)
    return df

def evaluate_model(model, X, y):
    probas = model.predict(X)
    score = roc_auc_score(y, probas)
    pred = [1 if item > 0.5 else 0 for item in probas]
    acc = accuracy_score(y, pred)
    print(f"auc = {score:.4}, acc={acc:.4}")

def build_simple_model(traindf, validdf):
    y_train = hp.get_single_column(traindf, 'income_level')
    y_test = hp.get_single_column(validdf, 'income_level')
    pipeline = Pipeline([
                        ('Var Dropper', hp.VarDropper(excl=['income_level'])),
                        ('Imputer', hp.Imputer()),
                        ('Encoder', hp.Encoder())])

    X_train_df = pipeline.transform(traindf)
    X_test_df = pipeline.transform(validdf)

    model = xgb.XGBClassifier()
    model.fit(X_train_df, y_train)
    evaluate_model(model, X_test_df, y_test)

if __name__ == '__main__':

    traindf, validdf, testdf = load_data()
    traindf = data_transform(traindf)
    validdf = data_transform(validdf)
    # testdf = data_transform(testdf)

    build_simple_model(traindf, validdf)
    # auc = 0.678, acc=0.9555

    # TODO: dummy transformation