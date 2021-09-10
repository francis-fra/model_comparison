import sys
import pandas as pd

lib_location = '/home/fra/Project/pyProj/zqlib/zq/dl'
sys.path.append(lib_location)
import kerasData as kd

data_location = '/home/fra/DataMart/datacentre/opendata/UCI/convertype/'

data = pd.read_csv(data_location + 'train_data.csv')
# print(data.head())

metadata = {}
TARGET_FEATURE_NAME = "Cover_Type"
TARGET_FEATURE_LABELS = ["0", "1", "2", "3", "4", "5", "6"]
NUMERIC_FEATURE_NAMES = [
    "Aspect",
    "Elevation",
    "Hillshade_3pm",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Horizontal_Distance_To_Fire_Points",
    "Horizontal_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Slope",
    "Vertical_Distance_To_Hydrology",
]
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "Soil_Type": list(data["Soil_Type"].unique()),
    "Wilderness_Area": list(data["Wilderness_Area"].unique()),
}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

CSV_HEADER = list(data.columns)
NUM_CLASSES = len(TARGET_FEATURE_LABELS)

metadata = kd.MetaData(TARGET_FEATURE_NAME,
                        CSV_HEADER,
                        NUMERIC_FEATURE_NAMES,
                        CATEGORICAL_FEATURE_NAMES,
                        FEATURE_NAMES,
                        CATEGORICAL_FEATURES_WITH_VOCABULARY,
                        NUM_CLASSES)

tmp = metadata.feature_names
tmp = metadata.categorical_value_dict
print(tmp)