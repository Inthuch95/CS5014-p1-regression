'''
Created on Mar 8, 2018

@author: User
'''
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.externals import joblib

X_train = np.loadtxt("../X_train")
y_train = np.loadtxt("../y_train")

linreg = LinearRegression()
linreg.fit(X_train, y_train)
joblib.dump(linreg, "../normal_linreg.pkl")