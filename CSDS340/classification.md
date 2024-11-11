## Classification Algorithms

### Naive Bayes

- **Probabilistic classifier**
- Based on Bayes' Theorem with the assumption that features are independent given the class. It computes the probability of each class given the feature values and classifies based on the highest probability.
- **Assumptions**:
  - Features are conditionally independent given the class label. This is rarely true in real-world data.
  - Assumes class conditional independence, meaning each feature contributes independently to the probability of a class label.
  
**Bayes' Theorem:**

$P(C_k \mid x) = \dfrac{P(x \mid C_k) \cdot P(C_k)}{P(x)}$

**For Gaussian Naive Bayes (continuous features):**

$P(x_i \mid C_k) = \dfrac{1}{\sqrt{2 \pi \sigma_k^2}} \exp \left( - \dfrac{(x_i - \mu_k)^2}{2 \sigma_k^2} \right)$

---

### Perceptron

- **Linear classifier**
- Single-layered neural network model that classifies data by finding a linear boundary. The Perceptron updates weights based on misclassified examples, iterating over the data until convergence or a set number of iterations.
- **Assumptions**:
  - Assumes that the data is linearly separable, meaning there exists a hyperplane that can perfectly separate the classes. If this isn't true, the Perceptron will not converge.
  - Typically used for binary classification, although multi-class versions exist.
  - Assumes no significant noise.

**Perceptron Update Rule:**

$w \leftarrow w + \eta \cdot (y - \hat{y}) \cdot x$

**Prediction Function (Linear):**

$\hat{y} = \text{sign}(w \cdot x + b)$

---

### Adaline and Gradient Descent

- **Linear classifier** (similar to Perceptron)
- Uses a continuous activation function (linear output) and updates weights using gradient descent. It minimizes the sum of the squared errors (MSE) instead of the misclassification errors.
- Can converge to a better solution due to the continuous output; useful for regression and classification.
- **Cons**: Sensitive to learning rate.
- **Assumptions**:
  - Assumes a linear relationship between input features and target variables.
  - Gradient descent works best when the errors are normally distributed.
  - Adaline also uses the MSE, assuming a continuous error surface that can be minimized by gradient descent.

**Activation Function (Linear):**

$z = w \cdot x + b$

**Mean Squared Error (MSE) Cost Function:**

$J(w) = \dfrac{1}{2} \sum (y - z)^2 = \dfrac{1}{2} \sum \left( y - (w \cdot x + b) \right)^2$

**Weight Update Rule using Gradient Descent:**

$w \leftarrow w + \eta \sum (y - z) \cdot x$

$b \leftarrow b + \eta \sum (y - z)$

---

### Support Vector Machines

- **Linear or nonlinear classifier**
- SVM finds the optimal hyperplane that separates data points of different classes with the maximum margin. The margin is defined as the distance between the hyperplane and the closest data points, which are called support vectors.
- SVM is effective in high-dimensional spaces and robust against overfitting.
- **Assumptions**:
  - Maximum margin principle: the best classification boundary is the one that maximizes the margin between the support vectors of each class.

**Decision Boundary:**

$f(x) = w \cdot x + b = 0$

**Optimization Objective:**

$\min_w \dfrac{1}{2} ||w||^2 \quad \text{subject to} \quad y_i(w \cdot x_i + b) \geq 1 \quad \forall i$

**For Soft-margin SVM:**

$\min_w \dfrac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i$

---

### Logistic Regression

- **Linear classifier** (though it uses a non-linear transformation)
- Logistic regression is used for binary classification tasks, predicting the probability that an instance belongs to a particular class. The model uses a linear combination of features and applies the sigmoid (logistic) function to map the output to a probability between 0 and 1.
- **Assumptions**:
  - Assumes a linear relationship between the input features and the log-odds of the target variable.
  - Assumes that the classes are separable in terms of the log-odds, but not necessarily linearly separable in feature space.
  - Assumes that there is little or no multicollinearity between the features.

**Sigmoid Function:**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
Where $z = w^T X + b$.

**Prediction Function:**

$$
P(y = 1 | X) = \sigma(w^T X + b) = \frac{1}{1 + e^{-(w^T X + b)}}
$$
If the predicted probability is greater than 0.5, the model classifies the instance as class 1, otherwise class 0.

**Log Loss (Cross-Entropy Loss):**

The model minimizes the log loss during training:
$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$
Where $y_i$ is the true label and $p_i$ is the predicted probability.

**Regularization (Optional):**

Logistic regression can be regularized to avoid overfitting:
- **L2 Regularization (Ridge)**: Penalizes the sum of squared coefficients.
- **L1 Regularization (Lasso)**: Penalizes the absolute values of the coefficients, encouraging sparsity.
- **Elastic Net**: Combines L1 and L2 regularization.



---

### Kernel Support Vector Machines

- **Nonlinear classifier**
- Extension of SVM that allows it to handle nonlinear data. The kernel trick is used to transform data into higher dimensions without explicitly calculating the transformation, enabling SVM to find a separating hyperplane even for complex datasets.
- Popular kernels: linear, polynomial, RBF (radial basis function)
- **Assumptions**:
  - Data is linearly separable in a higher-dimensional space via the kernel function.

**Kernel Trick (for nonlinear data):**

$f(x) = \sum \alpha_i y_i K(x_i, x_j) + b$

**Common Kernels:**

- **Linear Kernel**:

  $K(x_i, x_j) = x_i \cdot x_j$

- **Polynomial Kernel**:

  $K(x_i \cdot x_j + 1)^d$

- **RBF (Radial Basis Function)**:

  $K(x_i, x_j) = \exp \left( - \gamma ||x_i - x_j||^2 \right)$

---

### Decision Trees

- **Tree-based classifier**
- Decision trees split data by recursively selecting features and thresholds that result in the best separation between classes. They do this by using criteria such as entropy, information gain, or Gini impurity. At each node, it makes a decision that leads to a classification at the leaf nodes.
- **Cons**: Prone to overfitting, especially with deep trees. You can avoid this by limiting the maximum depth.
- **Assumptions**:
  - Assumes irrelevant features are ignored and will not be used in splits, since the splitting criteria select only the most informative features.
  - Assumes splits can be made along one feature at a time; not ideal for data that requires non-axis-aligned boundaries.
  - Assumes splits are made independently at each level of the tree.

**Information Gain:**

$\text{IG}(D, A) = H(D) - \sum_{v \in A} \dfrac{|D_v|}{|D|} H(D_v)$

**Entropy:**

$H(D) = - \sum_{i=1}^{k} p_i \log_2(p_i)$

---

### K Nearest Neighbors (KNN)

- **Instance-based classifier**
- Lazy learning algorithm that doesn't build a model during training. Instead, it classifies new points by looking at the "K" closest training examples and choosing the most common class among them. This makes it computationally expensive at prediction time. It is also sensitive to the choice of "k" and requires feature scaling for better performance.
- **Assumptions**:
  - Local homogeneity (points closer to each other are more similar).
  - Assumes feature importance is equal.
  - Uses Euclidean distance, which is an assumption in itself.
  - Feature scaling is necessary since features with larger ranges can dominate the distance calculations, violating the assumption that all features contribute equally.

**Distance Metric (Euclidean Distance):**

$d(x_i, x_j) = \sqrt{ \sum_{k=1}^{n} (x_{ik} - x_{jk})^2 }$

**Classification Decision:**

$\hat{y} = \text{mode}\left( \{ y_{i_1}, y_{i_2}, \dots, y_{i_k} \} \right)$
