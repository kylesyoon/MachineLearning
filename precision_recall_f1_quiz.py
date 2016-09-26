# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

from sklearn import cross_validation
from sklearn.metrics import f1_score

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y)

clf1 = DecisionTreeClassifier()
clf1.fit(x_train, y_train)

recall_tree = recall(y_test, clf1.predict(x_test))
precision_tree = precision(y_test ,clf1.predict(x_test))
f1_score_tree = f1_score(y_test, clf1.predict(x_test))

print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recall_tree, precision_tree)
print "Decision Tree F1 Score: {:.2f}".format(f1_score_tree)

clf2 = GaussianNB()
clf2.fit(x_train, y_train)
recall_gaussian = recall(y_test, clf2.predict(x_test))
precision_gaussian = precision(y_test ,clf2.predict(x_test))
f1_score_gaussian = f1_score(y_test, clf2.predict(x_test))

print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall_gaussian, precision_gaussian)
print "GaussianNB Tree F1 Score: {:.2f}".format(f1_score_gaussian)
