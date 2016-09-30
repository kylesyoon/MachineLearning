# ERROR DUE TO BIAS

# In this exercise we'll examine a learner which has high bias, and is incapable of
# learning the patterns in the data.
# Use the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error.

from sklearn.linear_model import LinearRegression
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.cross_validation import KFold
import numpy as np

# Set the learning curve parameters; you'll need this for learning_curves
size = 1000
cv = KFold(size, shuffle=True)
score = make_scorer(explained_variance_score)

# Create a series of data that forces a learner to have high bias
X = np.reshape(np.random.normal(scale=2,size=size),(-1,1))
y = np.array([[1 - 2*x[0] +x[0]**2] for x in X])

def plot_curve():
    reg = LinearRegression()
    reg.fit(X, y)

    print "Regressor score: {:.4f}".format(reg.score(X, y))

    # TODO: Use learning_curve imported above to create learning curves for both the
    #       training data and testing data. You'll need 'size', 'cv' and 'score' from above.
    training_sizes, training_scores, testing_scores = learning_curve(reg, X, y, train_sizes = np.linspace(0.1, 1.0, 10), cv = cv, scoring = score)
    print training_sizes
    print training_scores
    print testing_scores

    # TODO: Plot the training curves and the testing curves
    #       Use plt.plot twice -- one for each score. Be sure to give them labels!
    plt.plot(training_scores, 'bo')
    plt.plot(testing_scores, 'r+')
    # Plot aesthetics
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Curve Score")
    plt.xlabel("Training Points")
    plt.legend(bbox_to_anchor=(1.1, 1.1))
    plt.show()

#plot_curve()

# ERROR DUE TO VARIANCE

# In this exercise we'll examine a learner which has high variance, and tries to learn
# nonexistant patterns in the data.
# Use the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error.

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import KFold
from sklearn.metrics import explained_variance_score, make_scorer
import numpy as np

# Set the learning curve parameters; you'll need this for learning_curves
size = 1000
cv = KFold(size,shuffle=True)
score = make_scorer(explained_variance_score)

# Create a series of data that forces a learner to have high variance
X = np.round(np.reshape(np.random.normal(scale=5,size=2*size),(-1,2)),2)
y = np.array([[np.sin(x[0]+np.sin(x[1]))] for x in X])

def plot_curve2():
    reg = DecisionTreeRegressor()
    reg.fit(X,y)
    print "Regressor score: {:.4f}".format(reg.score(X,y))

    # TODO: Use learning_curve imported above to create learning curves for both the
    #       training data and testing data. You'll need 'size', 'cv' and 'score' from above.

    training_sizes, training_scores, testing_scores = learning_curve(reg, X, y, train_sizes = np.linspace(0.1, 1.0, 10), cv = cv, scoring = score)

    # TODO: Plot the training curves and the testing curves
    #       Use plt.plot twice -- one for each score. Be sure to give them labels!
    plt.plot(training_scores, 'bo')
    plt.plot(testing_scores, 'r+')

    # Plot aesthetics
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Curve Score")
    plt.xlabel("Training Points")
    plt.legend(bbox_to_anchor=(1.1, 1.1))
    plt.show()

plot_curve2()
