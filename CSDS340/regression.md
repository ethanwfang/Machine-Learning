
## Regression Models

Regression models are a type of statistical model used to predict a continuous outcome variable based on one or more input features. These models estimate relationships between input and output features to predict values for new data points.

### Linear Regression

This model assumes a linear relationship between the dependent and independent variables. It finds the line of best fit by minimizing the mean squared error (MSE). 

Linear regression that uses the mean-squared error as the loss function is referred to as ordinary least squares (OLS). 

Assumptions:  
1. Linearity  
2. Independence  
3. Homoscedasticity: variance of residuals is constant  
4. Normality: residuals are randomly distributed  

### Random Forest Regression
Ensemble learning method that builds multiple decision trees during training and averages their predictions for better accuracy and robustness. The final prediction is the average of the predictions from all trees.

Advantages: handles non-linear relationships effectively, robust to outliers, reduces overfitting compared to individual decision trees, can model complex patterns

Assumptions: 
1. No linear relationship  
2. Independence of observations (data is not highly correlated)
3. Features are somewaht important to predict the target variable  

### Ridge Regression 

This is a type of linear regression that includes regularization to prevent overfitting and improve the stability of the model. Specifically, ridge regression adds an L2 penalty to the loss function to shrink the model coefficients towards zero by adding a constraint on their magnitudes. This helps when the data has mluticollinearity or when the number of features is much larger than the number of observations.

$J(β) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$

where
- $y_i$ is the actual value of the response variable
- $\hat{y}_i$ is the predicted value
- $\beta_j$ are the regression coefficients
- $\lambda$ is a hyperparameter that controls the strength of the regularization.


$$
\hat{β} = (X^T X + \lambda I)^{-1} X^T y
$$

The key difference between ridge and OLS is the addition of the penalty term, which is the sum of the squared coefficients.



### Evaluation Metrics for Regression

- **Mean Absolute Error (MAE)**: Calculates the average absolute difference between the actual and predicted values. 

  $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

MAE represents the average magnitude of the errors without considering their direction (positive or negative). It's easy to interpret because it is in the same units as the original data.

- **Mean Squared Error (MSE)**: Measures the average of the squared differences between actual and predicted values. By squaring the errors, it gives more weight to larger deviations, making it more sensitive to outliers. This metric is useful when large errors are particularly undesirable.

  $
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $


- **R-Squared (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. R² values range from 0 to 1, where 1 indicates a perfect fit and 0 means no explanatory power.

  $
  R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
  $

- **Adjusted R-Squared**: A modified version of R² that adjusts for the number of predictors in the model, helping prevent overfitting. It’s especially useful for comparing models with different numbers of predictors.

  $
  \text{Adjusted } R^2 = 1 - \left(1 - R^2\right) \frac{n - 1}{n - p - 1}
  $

  where \( n \) is the number of observations, and \( p \) is the number of predictors.

- **Mean Absolute Percentage Error (MAPE)**: Calculates the average of the absolute percentage errors, providing a relative measure of error. It’s useful for understanding error in percentage terms but can be sensitive to small values in the denominator.

  $
  \text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100
  $

