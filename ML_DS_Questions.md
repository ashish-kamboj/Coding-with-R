## Questions

- Difference between Statistical Modeling and Machine Learning
- **What is selection Bias?**
  - Selection Bias is a clinical trials is a result of the sample group not representing the entire target population.
- What is normal distribution
- **Covariance Vs Correlation**
  - **Covariance** - indicates the direction of the linear relationship between variable
  - **Correlation** - measures both the strength and direction of the linear relationship between two variables
  - correlation values are standardized whereas, covariance values are not
  - Covariance is affected by the change in scale, i.e. if all the value of one variable is multiplied by a constant and all the value of another variable are multiplied, by a similar or different constant, then the covariance is changed. As against this, correlation is not influenced by the change in scale.
  - Correlation is dimensionless, i.e. it is a unit-free measure of the relationship between variables. Unlike covariance, where the value is obtained by the product of the units of the two variables.

- Difference between Z-test, T-test and F-test
- Why do we take sum of square in Linear Regression?
- Difference betweence correlation and VIF
- If two variables are correlated, How to decide which one to remove?

- **How does Variance Inflation Factor(VIF) Work?**
  - Regress each of the independent varables w.r.t rest of the independent variables in the model and calculate the R2 for each. Using R2 we can calculate the VIF of each variable i.e. VIF=1/(1-R2). Higher R2 value of independent variable corresponds to the high correlation, means the variable need to be removed.
  
- **Effect of Multicollinearity**
  - Moderate multicollinearity may not be problematic. However, severe multicollinearity is a problem because it can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable and difficult to interpret. Multicollinearity saps the statistical power of the analysis, can cause the coefficients to switch signs, and makes it more difficult to specify the correct model.
  
- **What is PCA?**
  - Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.
 
- **How does Principal Component Analysis(PCA) works?** [(here)](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
  - Standardization - standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis.
  - Covariance Matrix computation - to understand how the variables of the input data set are varying from the mean with respect to each other, or in other words, to see if there is any relationship between them. Because sometimes, variables are highly correlated in such a way that they contain redundant information. So, in order to identify these correlations, we compute the covariance matrix.
  - Compute the Eigenvectors and Eigenvalues of the covariance matrix to identify the principal components
  - Feature vector
  - Recast the data along the principal component axes

- **Mathematics of Eigenvalues and Eigen vectors**
  - A- XI (Where A is the matrix, X Is lambda, I is identity matrix)
  - take the determinent of A-XI i.e. det(A-XI) = 0
  - Solve of lambda, which wich will give the Eigen vaues
  - Using Eigen values get the Eigen vector (having unit length)
  
- **How PCA take cares of multicollinearity**
  - As Principle components are orthogonal to each other which helps in to get rid of multicollineraity
  
- Why the Principal components are othogonal to each other?
- Though each principal components are orthogonal (i.e. prevents multicollinearity) but still in principal components we have correlated variables. Do we not have to remove those?
- Difference between PCA and Random Forest for feature selection.

- **How can we overcome Overfitting in a Regression Model?**
  - Reduce the model complexity
  - Regularization
    - **Ridge Regression(L2 Regularization)**
      - It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
      - It reduces the model complexity by coefficient shrinkage.
    - **Lasso Regression(L1 Regularization)**
      - It is generally used when we have more number of features, because it automatically does feature selection.

- How to explain gain and lift to business person?
- How you will define Precision
- **How to handle class imbalance problem?**
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

- **What are the shortcomings of ROC AUC curve?**
  - https://www.kaggle.com/lct14558/imbalanced-data-why-you-should-not-use-roc-curve
  - https://stats.stackexchange.com/questions/193138/roc-curve-drawbacks

- **When to use Logistic Regression vs SVM? or Differences between Logistic Regression and SVM**
  - Logistic loss diverges faster than hinge loss. So, in general, it will be more sensitive to outliers.
  - Logistic loss does not go to zero even if the point is classified sufficiently confidently. This might lead to minor degradation in accuracy.
  - SVM try to maximize the margin between the closest support vectors while LR the posterior class probability. Thus, SVM find a solution which is as fare as possible for the two categories while LR has not this property.
  - LR is more sensitive to outliers than SVM because the cost function of LR diverges faster than those of SVM.
  - Logistic Regression produces probabilistic values while SVM produces 1 or 0. So in a few words LR makes not absolute prediction and it does not assume data is enough to give a final decision. This maybe be good property when what we want is an estimation or we do not have high confidence into data.
  
  **Note:**
  - SVM tries to find the widest possible separating margin, while Logistic Regression optimizes the log likelihood function, with probabilities modeled by the sigmoid function.
  - SVM extends by using kernel tricks, transforming datasets into rich features space, so that complex problems can be still dealt with in the same “linear” fashion in the lifted hyper space.

- Working of multiclass classification

- **What is Gradient Descent**
  - Gradient descent is used to minimize the cost function or any other function
  
- Working of Gradient Descent
- How to perform clustering on large dataset?
- How the recommendation system work if I don't like/dislike the any movies (in case of Netflix), just simply watch the movies there then, How the rating will be given (means the User vector is defined)?
- Hyrid recommendation system
- How to measure the performance of recommendation system?

- **ALS (alternating Least Squares) for Collaborative Filtering (algorithm for recommendation)**
  - ALS recommender is a matrix factorization algorithm that uses Alternating Least Squares with Weighted-Lamda-Regularization (ALS-WR). It factors the user to item matrix A into the user-to-feature matrix U and the item-to-feature matrix M: It runs the ALS algorithm in a parallel fashion. The ALS algorithm should uncover the latent factors that explain the observed user to item ratings and tries to find optimal factor weights to minimize the least squares between predicted and actual ratings.
  - [How does Netflix recommend movies? Matrix Factorization](https://www.youtube.com/watch?v=ZspR5PZemcs)
  - [Prototyping a Recommender System Step by Step Part 2: Alternating Least Square (ALS) Matrix Factorization in Collaborative Filtering](https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1)
  - [Explicit Matrix Factorization: ALS, SGD, and All That Jazz](https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea)
  - [Alternating Least Squares (ALS) Spark ML](https://www.elenacuoco.com/2016/12/22/alternating-least-squares-als-spark-ml/?cn-reloaded=1)

- **Alternating least squares algorithm (ALS)**
  - It holds one part of a model constant and doing OLS on the rest; then assuming the OLS coefficients and holding that part of the model constant to do OLS on the part of the model that was held constant the first time. The process is repeated until it converges. It's a way of breaking complex estimation or optimizations into linear pieces that can be used to iterate to an answer.

- How to visualize the data pattern in dataset having large number of featues (Know this will help in type of algorithm to fit)?

<br></br>
### R Interview questions
- How to read multiple files in one command in R?
- Difference between sapply(), lapply and tapply()





