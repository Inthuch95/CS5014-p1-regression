'''
Created on 16 Mar 2018
'''
from sklearn.externals import joblib
from sklearn.metrics.regression import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures 
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def save_fig(fig_name, tight_layout=True):
    path = os.path.join("../Figures", fig_name + ".png")
    print("Saving figure", fig_name)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

print("Loading testing data")
X_test = np.loadtxt("../prepared_data/X_test.txt")
y_test = np.loadtxt("../prepared_data/y_test.txt")

# transform the data
full_pipeline = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2, include_bias=True))
    ])
X_test = full_pipeline.fit_transform(X_test)
print("Loading Polynomial ridge regression model")
model = joblib.load("../models/poly_ridge_reg.pkl")

# get the error of testing set
print("")
y_hat = model.predict(X_test)
mse = mean_squared_error(y_test, y_hat)
rmse = np.sqrt(mse)
print("Test set result")
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R^2: ", model.score(X_test, y_test))

# plot final model prediction
x = np.array([i+1 for i in range(len(y_test))])
plt.title("Polynomial Ridge Regression")
plt.xlabel("Index")
plt.ylabel("Appliances")
plt.scatter(x, y_test, color="blue", label="Observed")
plt.plot(x, y_hat, color="red", alpha=0.5, label="Predicted")
plt.legend()
save_fig("final_model")
plt.show()