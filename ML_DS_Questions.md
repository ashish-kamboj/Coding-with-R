### Q. Difference between Statistical Modeling and Machine Learning

---

### Q. What is selection Bias?
Selection Bias is a clinical trials is a result of the sample group not representing the entire target population.

---

### Q. Probability vs Likelihood [(here)](https://medium.com/swlh/probability-vs-likelihood-cdac534bf523)
  - Probability is used to finding the chance of occurrence of a particular situation, whereas Likelihood is used to generally maximizing the chances of a particular situation to occur.
  - [What is the difference between “likelihood” and “probability”?](https://stats.stackexchange.com/questions/2641/what-is-the-difference-between-likelihood-and-probability)

---

### Q. Types of Distribution [(here)](https://www.analyticsvidhya.com/blog/2017/09/6-probability-distributions-data-science/)

---

### Q. What is normal distribution

---

### Q. Baye's Theorem** [(here)](https://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification)
  - Independent Events
  - Dependent Events
  -   - conditional Probability  :: P(A|B) = P(A∩B)/P(B)
					<br>OR</br>
			                          P(B|A) = P(B∩A)/P(A)	
  - Baye's Theorem :: P(A|B) = P(B|A)*P(A)/P(B)	<br>
		   P(A|B) = Posterior Probability <br>
		   P(B|A) = Likelihood <br>
		   P(A) = Prior probability <br>
		   P(B) = Marginal Probability
---	

### Q. Covariance Vs Correlation
  - **Covariance** - indicates the direction of the linear relationship between variable
  - **Correlation** - measures both the strength and direction of the linear relationship between two variables
  - correlation values are standardized whereas, covariance values are not
  - Covariance is affected by the change in scale, i.e. if all the value of one variable is multiplied by a constant and all the value of another variable are multiplied, by a similar or different constant, then the covariance is changed. As against this, correlation is not influenced by the change in scale.
  - Correlation is dimensionless, i.e. it is a unit-free measure of the relationship between variables. Unlike covariance, where the value is obtained by the product of the units of the two variables.
---

### Q. Difference between Z-test, T-test and F-test
  - [Statistical Tests — When to use Which ?](https://towardsdatascience.com/statistical-tests-when-to-use-which-704557554740)
  - [Hypothesis testing; z test, t-test. f-test](https://www.slideshare.net/shakehandwithlife/hypothesis-testing-z-test-ttest-ftest)
---

### Q. Missing Values Imputation
 - Do Nothing
 - Imputation Using (Mean/Median) Values
 - Imputation Using (Most Frequent) or (Zero/Constant) Values
 - Imputation Using k-NN
 - Imputation Using Multivariate Imputation by Chained Equation (MICE)
 - Imputation Using Deep Learning (Datawig)

---

### Q. Techniques for features or variable selection
 - **Univariate Selection** - Statistical tests can be used to select those features that have the strongest relationship with the output variable.  For example the **ANOVA F-value** method is appropriate for numerical inputs and categorical data, as we see in the Pima dataset. This can be used via the **f_classif()** function in  **SelectKBest** class of scikit-learn library library that can be used with a suite of different statistical tests to select a specific number of features.
 - **Recursive Feature Elimination** - The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain.It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.
 - **Principal Component Analysis** - Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form.
 - **Feature Importance** - Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.

---

### Q. Dimensionality Reduction algorithms
 - Principal Component Analysis
 - Singular Value Decomposition
 - Linear Discriminant Analysis
 - Isomap Embedding
 - Locally Linear Embedding
 - Modified Locally Linear Embedding

---

### Q. Why do we take sum of square in Linear Regression?

---

### Q. Difference betweence correlation and VIF

---

### Q. If two variables are correlated, How to decide which one to remove?

---

### Q. How does Variance Inflation Factor(VIF) Work?
Regress each of the independent varables w.r.t rest of the independent variables in the model and calculate the R2 for each. Using R2 we can calculate the VIF of each variable i.e. VIF=1/(1-R2). Higher R2 value of independent variable corresponds to the high correlation, means the variable need to be removed.

---

### Q. Effect of Multicollinearity
Moderate multicollinearity may not be problematic. However, severe multicollinearity is a problem because it can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable and difficult to interpret. Multicollinearity saps the statistical power of the analysis, can cause the coefficients to switch signs, and makes it more difficult to specify the correct model.

---

### Q. What is PCA?
Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

---

### Q. How does Principal Component Analysis(PCA) works? [(here)](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
  - Standardization - standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis.
  - Covariance Matrix computation - to understand how the variables of the input data set are varying from the mean with respect to each other, or in other words, to see if there is any relationship between them. Because sometimes, variables are highly correlated in such a way that they contain redundant information. So, in order to identify these correlations, we compute the covariance matrix.
  - Compute the Eigenvectors and Eigenvalues of the covariance matrix to identify the principal components
  - Feature vector
  - Recast the data along the principal component axes
  - [How to Calculate Principal Component Analysis (PCA) from Scratch in Python](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)

---

### Q. Mathematics of Eigenvalues and Eigen vectors
  - A- XI (Where A is the matrix, X Is lambda, I is identity matrix)
  - take the determinent of A-XI i.e. det(A-XI) = 0
  - Solve of lambda, which wich will give the Eigen values
  - Using Eigen values get the Eigen vector (having unit length)

---

### Q. How PCA take cares of multicollinearity**
As Principle components are orthogonal to each other which helps in to get rid of multicollineraity

---

### Q. Why the Principal components are othogonal to each other?
Though each principal components are orthogonal (i.e. prevents multicollinearity) but still in principal components we have correlated variables. Do we not have to remove those?

---

### Q. Difference between PCA and Random Forest for feature selection.

---

### Q. How can we overcome Overfitting in a Regression Model?**
  - Reduce the model complexity
  - Regularization
    - **Ridge Regression(L2 Regularization)**
      - It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
      - It reduces the model complexity by coefficient shrinkage.
    - **Lasso Regression(L1 Regularization)**
      - It is generally used when we have more number of features, because it automatically does feature selection.

---

### Q. How to prevent overfitting [(here)](https://elitedatascience.com/overfitting-in-machine-learning)
   - Cross-validation
   - Train with more data
   - Remove features
   - Early stopping
   - Regularization
   - Ensembling

---

### Q. How to explain gain and lift to business person?

---

### Q. How you will define Precision

---

### Q. How to handle class imbalance problem?
  - Get more data
  - Try different performance matrix
    - Confusion Matrix
    - Precision
    - Recall
    - F1-Score
    - Kappa
    - Area Under ROC curve
  - Data Resampling
    - Undersampling
    - Oversampling
  - Generate synthetic data
  - Use different algorithms for classification
  - Try Penalized models
    - penalized-SVM
    - penalized LDA
  - Try different techniques
    - Anomaly detection
    - Change detection

---

### Q. What are the shortcomings of ROC AUC curve?**
  - https://www.kaggle.com/lct14558/imbalanced-data-why-you-should-not-use-roc-curve
  - https://stats.stackexchange.com/questions/193138/roc-curve-drawbacks

---

### Q. When to use Logistic Regression vs SVM? or Differences between Logistic Regression and SVM**
  - Logistic loss diverges faster than hinge loss. So, in general, it will be more sensitive to outliers.
  - Logistic loss does not go to zero even if the point is classified sufficiently confidently. This might lead to minor degradation in accuracy.
  - SVM try to maximize the margin between the closest support vectors while LR the posterior class probability. Thus, SVM find a solution which is as fare as possible for the two categories while LR has not this property.
  - LR is more sensitive to outliers than SVM because the cost function of LR diverges faster than those of SVM.
  - Logistic Regression produces probabilistic values while SVM produces 1 or 0. So in a few words LR makes not absolute prediction and it does not assume data is enough to give a final decision. This maybe be good property when what we want is an estimation or we do not have high confidence into data.
  
  > **_Note:_**
  - SVM tries to find the widest possible separating margin, while Logistic Regression optimizes the log likelihood function, with probabilities modeled by the sigmoid function.
  - SVM extends by using kernel tricks, transforming datasets into rich features space, so that complex problems can be still dealt with in the same “linear” fashion in the lifted hyper space.

---

### Q. Logistic Regression Vs GLM 
Logistic Regression is the special case of GLM with  `distribution type=Bernoulli` and `LinkFunction=Logit`. Below are the various linear models we can run by changing the **distribution type** and **LinkFunction**

|Distribution Type    	  | LinkFunction | PredictFactor | ComponentModel     |
|-------------------------|--------------|---------------|--------------------|
|Normal                   | Identity     | Continuous    | Linear Regression  |
|Normal                   | Identity     | Categorical   | Anal. of Variance  |
|Normal                   | Identity     | Mixed         | Anal. of Covariance|
|Bernoulli                | Logit        | Mixed         | Logistic Regression|
|Poisson                  | Log          | Categorical   | Log-linear         |
|Poisson                  | Log          | Mixed         | Poisson Regression |
|Gamma(Positive ontinuous)| Log          | Mixed         | Gamma Regression   |

---

### Q. Working of multiclass classification

---

### Q. What is Gradient Descent
Gradient descent is used to minimize the cost function or any other function

---

### Q. What is the difference between Gradient Descent and Stochastic Gradient Descent? [(here)](https://datascience.stackexchange.com/questions/36450/what-is-the-difference-between-gradient-descent-and-stochastic-gradient-descent#:~:text=In%20Gradient%20Descent%20or%20Batch,of%20training%20data%20per%20epoch)

---

### Q. Working of Gradient Descent

---

### Q. Algorithms for clustering
 - **K-Means** - For datasets having numerical variables
 - **Mini-Batch K-Means** - Mini-Batch K-Means is a modified version of k-means that makes updates to the cluster centroids using mini-batches of samples rather than the entire dataset, which can make it faster for large datasets, and perhaps more robust to statistical noise.
 - **Hierarchical clustering**
 - **K-Mode** - For datasets having categorical variables
 - **k-Prototype** - It combines the k-modes and k-means algorithms and is able to cluster mixed numerical and categorical variables
 - **PAM (Partitioning Around Medoids) or K-Medoids** - For datasets having both numerical and categorical variables

---

### Q. How to find optimal value of k (or number of clusters) in clustering?
 - Elbow method
 - Silhouette coefficient 

---

### Q. How to compute the mean Silhouette Coefficient of all samples?
The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.

---

### Q. How will you check if the segmenting is good or whether you need to use different factors for segmenting(K-Means)?
One really easy way is to see if the .are logically correct. For example, we can check if states in a similar geography and economic situation are clustered together or not. You can also run hypothesis test to check whether the population of different clusters is significantly different or not, If you look at the data, you can see that some specific customers or some specific states should be grouped together.

---

### Q. What are the benefits of Hierarchical Clustering over K-Means clustering? What are the disadvantages?
Hierarchical clustering generally produces better clusters, but is more computationallyintensive.

---

### Q. How to calculate Gower’s Distance using Python [(here)](https://medium.com/analytics-vidhya/concept-of-gowers-distance-and-it-s-application-using-python-b08cf6139ac2)

---

### Q. How to perform clustering on large dataset?

---

### Q. How the recommendation system work if I don't like/dislike the any movies (in case of Netflix), just simply watch the movies there then, How the rating will be given (means the User vector is defined)?

---

### Q. Hyrid recommendation system

---

### Q. How to measure the performance of recommendation system?

---

### Q. ALS (alternating Least Squares) for Collaborative Filtering (algorithm for recommendation)
  - ALS recommender is a matrix factorization algorithm that uses Alternating Least Squares with Weighted-Lamda-Regularization (ALS-WR). It factors the user to item matrix A into the user-to-feature matrix U and the item-to-feature matrix M: It runs the ALS algorithm in a parallel fashion. The ALS algorithm should uncover the latent factors that explain the observed user to item ratings and tries to find optimal factor weights to minimize the least squares between predicted and actual ratings.
  - [How does Netflix recommend movies? Matrix Factorization](https://www.youtube.com/watch?v=ZspR5PZemcs)
  - [Prototyping a Recommender System Step by Step Part 2: Alternating Least Square (ALS) Matrix Factorization in Collaborative Filtering](https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1)
  - [Explicit Matrix Factorization: ALS, SGD, and All That Jazz](https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea)
  - [ALS Implicit Collaborative Filtering](https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe)
  - [Alternating Least Squares (ALS) Spark ML](https://www.elenacuoco.com/2016/12/22/alternating-least-squares-als-spark-ml/?cn-reloaded=1)

---

### Q. Alternating least squares algorithm (ALS)**
It holds one part of a model constant and doing OLS on the rest; then assuming the OLS coefficients and holding that part of the model constant to do OLS on the part of the model that was held constant the first time. The process is repeated until it converges. It's a way of breaking complex estimation or optimizations into linear pieces that can be used to iterate to an answer.

---

### Q. Hyperparameters to tune in Logistic Regression**
  - Logistic regression does not really have any critical hyperparameters to tune.
    - Sometimes, you can see useful differences in performance or convergence with different solvers (solver).
      - **solver** in [‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]
    - Regularization (penalty) can sometimes be helpful.
      - **penalty** in [‘none’, ‘l1’, ‘l2’, ‘elasticnet’]

---

### Q. Hyperparameters to tune in Random Forest**
  - **n_estimators** = number of trees in the foreset
  - **max_features** = max number of features considered for splitting a node
  - **max_depth** = max number of levels in each decision tree
  - **min_samples_split** = min number of data points placed in a node before the node is split
  - **min_samples_leaf** = min number of data points allowed in a leaf node
  - **bootstrap** = method for sampling data points (with or without replacement)

---

### Q. Hyperparameters to tune in XGBoost** [(here)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

---

### Q. How to visualize the data pattern in dataset having large number of featues (Know this will help in type of algorithm to fit)?

---
<br></br>

---
## Courses
1. Google Machine Learning crash course [(here)](https://developers.google.com/machine-learning/crash-course/ml-intro)
2. Practical Deep Learning for Coders [(here)](https://course.fast.ai/)
3. fast.ai courses [(here)](https://www.fast.ai/)






