import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Ridge regression model with a specific alpha (regularization strength)
ridge_reg = Ridge(alpha=1.0)

ridge_reg.fit(X_train, y_train)

y_pred = ridge_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Coefficients: {ridge_reg.coef_}")
print(f"Intercept: {ridge_reg.intercept_}")

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.plot(X_test, y_pred, color='red')
plt.title('Ridge Regression Results')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
