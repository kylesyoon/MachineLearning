# MachineLearning
Notes from the Udacity MLND

# Precision and Recall
Confusion Matrix
```
[TP FN]
[FP TN]
```

`Recall = True Positive / (True Positive + False Negative)`
`Precision = True Positive / (True Positive + False Positive)`

# F1 Score
The F1 score can be interpreted as a weighted average of the prevision and recall. ([0, 1])
```
F1 = 2 * (precision * recall) / (precision + recall)
```

# R2 Score
Computed the coefficient of determination of predictions for true values.
The default scoring method for regression learners in scikit-learn.

# Explained Variance Score

# Bias-variance Dilemma
High Bias
- Pays little attention to data
- Oversimplified
- High error on training set (low r^2, large SSE)
- too few features

High Variance
- Pays too much attention to data (does not generalize well)
- Overfits
- Much high error on test set than training set
- too many features, carefully optimized performance on training data

=> You want few features as possible. Large r^2, low SSE.

# Curse of Dimensionality
As the number of features or dimensions grows, the amount of data we need to generalize accurately grows EXPONENTIALLY!

# Cross Validation
Divide the available data in to K number of groups. Each group takes a turn being the test data. The result is averaged. This allows all the data to be training and testing data.
