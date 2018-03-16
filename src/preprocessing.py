'''
Created on Mar 8, 2018
'''
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from util.df_selector import DataFrameSelector
from util.attributes_util import add_attributes_from_dates
from util.label import MyLabelBinarizer
import pandas as pd
import numpy as np

print("Loading energy dataset")
df = pd.read_csv("../energydata_complete.csv")

# add extra attributes
print("Adding extra attributes")
df = add_attributes_from_dates(df)

# remove unimportant predictors
print("Dropping unimportant attributes")
df.drop(["rv1","rv2","date"], axis=1, inplace=True)

# split training set and testing set (80/20 ratio)
print("Splitting train/test set")
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_set.drop("Appliances", axis=1)
y_train = train_set["Appliances"].copy()
X_test = test_set.drop("Appliances", axis=1)
y_test = test_set["Appliances"]

# create transformation pipeline
day_attrib = ["day_of_week"]
status_attrib = ["week_status"]
num_attribs = list(df.drop(["day_of_week","week_status","Appliances"], axis=1))
num_pipeline = Pipeline([
        ("selector", DataFrameSelector(num_attribs)),
        ("std_scaler", StandardScaler()),
    ])
day_pipeline = Pipeline([
        ("selector", DataFrameSelector(day_attrib)),
        ("label_binarizer", MyLabelBinarizer()),
    ])
status_pipeline = Pipeline([
        ("selector", DataFrameSelector(status_attrib)),
        ("label_binarizer", MyLabelBinarizer()),
    ])
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("day_pipeline", day_pipeline),
        ("cat_pipeline", status_pipeline),
    ])
print("Transforming inputs")
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.fit_transform(X_test)

# save pre-processed data into text files
print("Saving data")
np.savetxt("../prepared_data/X_train.txt", X_train)
np.savetxt("../prepared_data/y_train.txt", y_train)
np.savetxt("../prepared_data/X_test.txt", X_test)
np.savetxt("../prepared_data/y_test.txt", y_test)
print("Pre-processing completed")