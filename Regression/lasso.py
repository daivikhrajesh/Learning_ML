import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Lasso regression model with alpha=1.0
lasso_rig = Lasso(alpha=1.0)

# Fit the model on the scaled training data
lasso_rig.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = lasso_rig.predict(X_test_scaled)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Coefficients: {lasso_rig.coef_}")
print(f"Intercept: {lasso_rig.intercept_}")

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.plot(X_test, y_pred, color='red')
plt.title('Lasso Regression Results')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
