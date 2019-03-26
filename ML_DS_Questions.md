## Questions

- Difference between Statistical Modeling and Machine Learning
- What is normal distribution
- Difference between Z-test, T-test and F-test
- Why do we take sum of square in Linear Regression?
- Difference betweence correlation and VIF
- If two variables are correlated, How to decide which one to remove?

- **How does Variance Inflation Factor(VIF) Work?**
  - Regress each of the independent varables w.r.t rest of the independent variables in the model and calculate the R2 for each. Using R2 we can calculate the VIF of each variable i.e. VIF=1/(1-R2). Higher R2 value of independent variable corresponds to the high correlation, means the variable need to be removed.
  
- **Effect of Multicollinearity**
  - Moderate multicollinearity may not be problematic. However, severe multicollinearity is a problem because it can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable and difficult to interpret. Multicollinearity saps the statistical power of the analysis, can cause the coefficients to switch signs, and makes it more difficult to specify the correct model.
  
- How does Principal Component Analysis(PCA) works?
  
- **How PCA take cares of multicollinearity**
  - As Principle components are orthogonal to each other which helps in to get rid of multicollineraity
  
- Why the Principal components are othogonal to each other?
- Though each principal components are orthogonal (i.e. prevents multicollinearity) but still in principal components we have correlated variables. Do we not have to remove those?
- Difference between PCA and Random Forest for feature selection.

- **How can we overcome Overfitting in a Regression Model?**
  - Reduce the model complexity
  - Regularization
    - Ridge Regression(L2 Regularization)
      - It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
      - It reduces the model complexity by coefficient shrinkage.
      - It uses L2 regularization technique. (which I will discussed later in this article)
      - Cost function for Ridge regression
      https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/05213609/snip14.png
    - Lasso Regression(L1 Regularization)
      - 

- How to explain gain and lift to business person?
- How you will define Precision
- What are the shortcomings of ROC curve?
- **When to use Logistic Regression vs SVM? or Differences between Logistic Regression and SVM**
  - Logistic loss diverges faster than hinge loss. So, in general, it will be more sensitive to outliers.
  - Logistic loss does not go to zero even if the point is classified sufficiently confidently. This might lead to minor degradation in accuracy.
  - SVM try to maximize the margin between the closest support vectors while LR the posterior class probability. Thus, SVM find a solution which is as fare as possible for the two categories while LR has not this property.
  - LR is more sensitive to outliers than SVM because the cost function of LR diverges faster than those of SVM.
  - Logistic Regression produces probabilistic values while SVM produces 1 or 0. So in a few words LR makes not absolute prediction and it does not assume data is enough to give a final decision. This maybe be good property when what we want is an estimation or we do not have high confidence into data.
  
  **Note:**
  - SVM tries to find the widest possible separating margin, while Logistic Regression optimizes the log likelihood function, with probabilities modeled by the sigmoid function.
  - SVM extends by using kernel tricks, transforming datasets into rich features space, so that complex problems can be still dealt with in the same “linear” fashion in the lifted hyper space.

- **What is Gradient Descent**
  - Gradient descent is used to minimize the cost function or any other function

- How to perform clustering on large dataset?
- How the recommendation system work if I don't like/dislike the any movies (in case of Netflix), just simply watch the movies there then, How the rating will be given (means the User vector is defined)?
- Hyrid recommendation system
- How to measure the performance of recommendation system?

<br></br>
### R Interview questions
- How to read multiple files in one command in R?
- Difference between sapply(), lapply and tapply()





