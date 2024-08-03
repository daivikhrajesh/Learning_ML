# Simple LR

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(0)
X = 2*np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)


slope = model.coef_[0][0]
intercept = model.intercept_[0]

print(f"Coefficient (slope): {model.coef_[0][0]}")
print(f"Intercept: {model.intercept_[0]}")

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# Plot the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()

plt.text(1, 15, f'Slope: {slope:.2f}', fontsize=12, color='red')
plt.text(1, 14, f'Intercept: {intercept:.2f}', fontsize=12, color='red')

plt.show()
