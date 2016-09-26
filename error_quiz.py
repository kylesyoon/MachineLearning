import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error as mse

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y)

reg1 = DecisionTreeRegressor()
reg1.fit(x_train, y_train)
mean_absolute_error_tree = mae(reg1.predict(x_test), y_test)
mean_squared_error_tree = mse(reg1.predict(x_test), y_test)
print "Decision Tree mean absolute error: {:.2f}".format(mean_absolute_error_tree)
print "Decision Tree mean absolute error: {:.2f}".format(mean_squared_error_tree)

reg2 = LinearRegression()
reg2.fit(x_train, y_train)
mean_absolute_error_linear = mae(reg2.predict(x_test), y_test)
mean_squared_error_linear = mse(reg2.predict(x_test), y_test)
print "Linear regression mean absolute error: {:.2f}".format(mean_absolute_error_linear)
print "Linear regression mean absolute error: {:.2f}".format(mean_squared_error_linear)
