# Chapter 4: Training Models

## **Linear Regression Training**

Linear regression predicts outputs by fitting a line (or hyperplane) that minimizes the difference between predictions and true values.

### **Key Techniques**

1. **Normal Equation**:
    - A closed-form solution for finding model parameters  $\theta$ :
        
        $$
        \theta = (X^T X)^{-1} X^T y
        $$
        
    - Computational complexity:  $O(n^2)$  to  $O(n^3)$ .
    - Implementation in Scikit-learn:
        
        ```python
        from sklearn.linear_model import LinearRegression
        
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        ```
        
2. **Gradient Descent**:
    - Iterative optimization to minimize the cost function:
        
        $$
        J(\theta) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2
        $$
        
    - Update rule:
        
        $$
        \theta_j := \theta_j - \eta \frac{\partial J(\theta)}{\partial \theta_j}
        $$
        
    - Implementation:
        
        ```python
        from sklearn.linear_model import SGDRegressor
        
        sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
        sgd_reg.fit(X_train, y_train.ravel())
        ```
        

### **Types of Gradient Descent**

1. **Batch Gradient Descent**:
    - Uses the entire dataset to compute gradients.
    - Guarantees convergence but is computationally expensive.
2. **Stochastic Gradient Descent (SGD)**:
    - Uses a single random sample at each step:
        
        ```python
        learning_rate = 0.1
        for epoch in range(epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                theta -= learning_rate * gradients
        ```
        
3. **Mini-Batch Gradient Descent**:
    - Combines the benefits of Batch and SGD:
        
        ```python
        batch_size = 32
        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(m)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            for i in range(0, m, batch_size):
                xi = X_shuffled[i:i+batch_size]
                yi = y_shuffled[i:i+batch_size]
                gradients = 2/batch_size * xi.T.dot(xi.dot(theta) - yi)
                theta -= learning_rate * gradients
        
        ```
        

---

## **Overfitting and Underfitting**

Overfitting and underfitting are critical challenges in machine learning.

### **Causes and Solutions**

1. **Overfitting**:
    - Occurs when the model captures noise instead of patterns.
    - Solutions:
        - Regularization.
        - Cross-validation.
        - Early stopping.
2. **Underfitting**:
    - Occurs when the model is too simple to capture patterns.
    - Solutions:
        - Increase model complexity.
        - Add more features.

### **Bias-Variance Tradeoff**

- **Bias**: Error due to overly simplistic models.
- **Variance**: Error due to overly complex models.
- **Irreducible Error**: Noise inherent in the data.

---

## **Regularized Linear Models**

Regularization adds a penalty term to the cost function to constrain model complexity.

### **Key Methods**

1. **Ridge Regression (L2 Regularization)**:
    - Penalizes large weights:
        
        $$
        J(\theta) = \text{MSE}(\theta) + \alpha \frac{1}{2} \sum_{i=1}^n \theta_i^2
        $$
        
    - Implementation:
        
        ```python
        from sklearn.linear_model import Ridge
        
        ridge_reg = Ridge(alpha=1.0)
        ridge_reg.fit(X_train, y_train)
        ```
        
2. **Lasso Regression (L1 Regularization)**:
    - Shrinks some coefficients to zero:
        
        $$
        J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^n |\theta_i|
        $$
        
    - Implementation:
        
        ```python
        from sklearn.linear_model import Lasso
        
        lasso_reg = Lasso(alpha=0.1)
        lasso_reg.fit(X_train, y_train)
        ```
        
3. **Elastic Net**:
    - Combines Ridge and Lasso:
        
        $$
        J(\theta) = \text{MSE}(\theta) + r \alpha \sum_{i=1}^n |\theta_i| + \frac{1-r}{2} \alpha \sum_{i=1}^n \theta_i^2
        $$
        
    - Implementation:
        
        ```python
        from sklearn.linear_model import ElasticNet
        
        elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elastic_net.fit(X_train, y_train)
        
        ```
        

---

## **Logistic Regression Training**

Logistic regression is used for binary classification.

### **Key Concepts**

1. **Sigmoid Function**:
    - Maps inputs to probabilities between 0 and 1:
        
        $$
        \sigma(z) = \frac{1}{1 + e^{-z}}
        $$
        
2. **Decision Boundary**:
    - Predicts positive if  $\sigma(x) \geq 0.5$ , negative otherwise.
3. **Cost Function**:
    - Log loss:
        
        $$
        J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]
        $$
        
4. **Implementation**:
    
    ```python
    from sklearn.linear_model import LogisticRegression
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    ```
    

---

## **Softmax Regression**

Softmax regression generalizes logistic regression for multiclass classification.

### **Key Concepts**

1. **Softmax Function**:
    - Estimates probabilities for multiple classes:
        
        $$
        \hat{p}k = \frac{e^{\theta_k^T x}}{\sum{j=1}^K e^{\theta_j^T x}}
        $$
        
2. **Cross-Entropy Loss**:
    - Penalizes incorrect predictions:
        
        $$
        J(\Theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log(\hat{p}_k^{(i)})
        $$
        
3. **Implementation**:
    
    ```python
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    softmax_reg.fit(X_train, y_train)
    ```
    

---

## **Practical Tips**

- **Feature Scaling**: Normalize features for gradient descent to converge efficiently.
- **Learning Rate Tuning**: Use learning rate schedules or adaptive methods like Adam.
- **Regularization Strength ( $\alpha$ )**: Balance bias-variance tradeoff by tuning  $\alpha$ .

---

## **Conclusion**

This chapter provides a comprehensive foundation for training machine learning models. By mastering linear regression, gradient descent, regularization, and classification techniques like logistic and softmax regression, you'll be equipped to design and train models that generalize well to unseen data.

### **Key Takeaways**

- Use the Normal Equation for small datasets and gradient descent for large datasets.
- Regularization (Ridge, Lasso, Elastic Net) helps prevent overfitting.
- Logistic regression is ideal for binary classification, while softmax regression handles multiclass problems.
- Always balance bias and variance to achieve optimal performance.