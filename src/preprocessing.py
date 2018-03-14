'''
Created on Mar 8, 2018

@author: User
'''
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from df_selector import DataFrameSelector

def calculate_nsm(dt_obj):
    """Get the number of seconds until midnight."""
    tomorrow = dt_obj + timedelta(1)
    midnight = datetime(year=tomorrow.year, month=tomorrow.month, 
                        day=tomorrow.day, hour=0, minute=0, second=0)
    return (midnight - dt_obj).seconds

print("Loading energy dataset")
df = pd.read_csv("../energydata_complete.csv")
# add number of seconds to midnight column 
nsm_list = []
df["nsm"] = ""
print("Calculating nsm")
for index, row in df.iterrows():
    dt_str = row["date"]
    dt_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    nsm_list.append(calculate_nsm(dt_obj))
df["nsm"] = nsm_list
df.drop(["rv1","rv2","date"], axis=1, inplace=True)

# split training set and testing set (80/20 ratio)
print("Splitting train/test set")
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_set.drop("Appliances", axis=1)
# X_train = train_set[["nsm","lights","Press_mm_hg","RH_3","T5","T8"]]
y_train = train_set["Appliances"].copy()
X_test = test_set.drop("Appliances", axis=1)
# X_test = test_set[["nsm","lights","Press_mm_hg","RH_3","T5","T8"]]
y_test = test_set["Appliances"]

print("Saving data")
# save pre-processed data into text files
np.savetxt("../X_train.txt", X_train)
np.savetxt("../y_train.txt", y_train)
np.savetxt("../X_test.txt", X_test)
np.savetxt("../y_test.txt", y_test)
print("Pre-processing completed")