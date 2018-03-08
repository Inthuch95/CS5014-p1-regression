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

energy = pd.read_csv("../energydata_complete.csv")
nsm_list = []
energy["nsm"] = ""
for index, row in energy.iterrows():
    dt_str = row["date"]
    dt_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    nsm_list.append(calculate_nsm(dt_obj))
energy["nsm"] = nsm_list
energy.drop(["rv1","rv2","date"], axis=1, inplace=True)

train_set, test_set = train_test_split(energy, test_size=0.2, random_state=42)
X_train = train_set.drop("Appliances", axis=1)
y_train = train_set["Appliances"].copy()

attribs = list(X_train)
full_pipeline = Pipeline([
        ("selector", DataFrameSelector(list(X_train))),
        ("std_scaler", StandardScaler()),
    ])

X_train = full_pipeline.fit_transform(X_train)
np.savetxt("../X_train", X_train)
np.savetxt("../y_train", y_train)