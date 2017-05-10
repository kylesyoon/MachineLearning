# Machine Learning Notes

##  Foundations

### Precision and Recall
Confusion Matrix
```
[TP FN]
[FP TN]
```

`Recall = True Positive / (True Positive + False Negative)`
`Precision = True Positive / (True Positive + False Positive)`

### F1 Score
The F1 score can be interpreted as a weighted average of the precision and recall. ([0, 1])
```
F1 = 2 * (precision * recall) / (precision + recall)
```

### R2 Score
Computed the coefficient of determination of predictions for true values.
The default scoring method for regression learners in scikit-learn.

### Explained Variance Score

#### Bias-variance Dilemma
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

### Curse of Dimensionality
As the number of features or dimensions grows, the amount of data we need to generalize accurately grows EXPONENTIALLY!

### Cross Validation
Divide the available data in to K number of groups. Each group takes a turn being the test data. The result is averaged. This allows all the data to be training and testing data.

## Supervised Learning

### Gaussian Naive Bayes
- Assumes that a particular feature is *independent* of the value of any other features, given the class variable.
- Disregards correlation between features.
- Only requires a small number of training data. So works well with smaller data sets.
- It is oversimplified, so it is outperformed by many other algorithms when the data is large.
- Used in spam filter, document classification, text analysis.

### Decision Trees
- Choose the feature with most information gain (least entropy), move on and repeat.
- Only goes one level at a time, so the model isn't always the best.
- Because it splits data until there is an answer, it is prone to overfitting.
- Easy to use, can be represented well graphically.
- Implicitly does feature selection.
- Performs well on nonlinear relationships between features.
- Can handle missing values and various feature types.
[ID3 Algorithm for Decision Trees](ID3%20Algorithm%20for%20Decision%20Trees.pdf)

### Ensemble Methods
- Combining a set of weak learning to produce a learner with high accuracy.
	- A weak learning is a learner whose perfomance is only slightly better than random guessing.
	- The weak learner must not be too complex to avoid overfitting.
- Advantages:
	- Computationally efficient.
	- No dofficult parameters to set.
	- Versatile
- Disadvantages:
	- Susceptible to uniform noise.
	- Needs a sufficient amount of data for the weak learner to do better than random guessing.
[Intro to Boosting](Intro%20to%20Boosting.pdf)

### KNN
- Lazy learner
	- Does not compute a function to fit the training data before new data is received.
- The data itself is the function to which new instances are fit.
- Larger memory requirements, and longer computational times.
- Advantages in local-scale estimation and easy integration of additional training data.
- Expects that the data has locality and smoothness.
	- Data points that are close to one another in distance are expected to have similar value.
[Instance Based Learning](Instance%20Based%20Learning.pdf)

### Neural Networks
#### Perceptrons
- ∑w*x > θ  = y

Simple way to train the w, for a given incorrectly labeled x, is to
```
w' = w + x
```
Considering negative outputs
```
w' = w + (y-y_hat)x
```
Adding a learning rate n
```
w' = w' + n(y-y_hat)x
```

### Gradient Descent
- Makes the error function quadratic so that the derivative can be used to find the weight delta. Allows to minimize the error.

#### Sigmoid function
- Use the signoid function to make a smoothed thresholding function. Makes the perceptions nonlinear.
[Neural Networks](Neural%20Networks.pdf)
[Gradient Descent](Gradient%20Descent.pdf)

### Stochastic Gradient Descent (SGD)
- is a simple yet very efficient approach to discriminative learning of linear classifiers under convex loss functions such as (linear) Support Vector Machines and Logistic Regression.
- Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.
- SGD has been successfully applied to large-scale and sparse machine learning problems often encountered in text classification and natural language processing. Given that the data is sparse, the classifiers in this module easily scale to problems with more than 10^5 training examples and more than 10^5 features.
The advantages of Stochastic Gradient Descent are:
- Efficiency.
- Ease of implementation (lots of opportunities for code tuning).
The disadvantages of Stochastic Gradient Descent include:
- SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
- SGD is sensitive to feature scaling.

### SVM
The advantages of support vector machines are:
- Effective in high dimensional spaces.
- Still effective in cases where number of dimensions is greater than the number of samples.
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
The disadvantages of support vector machines include:
- If the number of features is much greater than the number of samples, the method is likely to give poor performances.
- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
[Kernel Methods and SVMs](Kernel_Methods_and_SVMs.pdf)

## Supervised Learning

### K-Means
- Need to define how many clusters. Can be a challenge.
- Initial placement can change the out come. (Hill climbing)
- Local minima

#### Single Linkage Clustering
- Define K = number of clusters
- Find two closest clusters and merge, repeat until there are K clusters
- hierarchical agglomerative cluster structure (tree)
*Running Time*
1. Repeat K times (n/2)
2. Look at all distances to find closest point (n^2)
= O(n^3)

### Soft Clustering (expectation maximization)
- Uses probability to cluster, allows points to be shared among clusters when probability of belonging to a specific cluster is low
- monotomically non-decreasing likelihood
- does not converge (infinitely configures), but practically does
- will not diverge
- can get stuck -> random restart
- works with any distribution

### Three Properties of Clustering
- Richness
- Scale-invariance
- Consistency

#### Impossibility Theorem by Kleinburg
It's impossible to achieve all three properties.
