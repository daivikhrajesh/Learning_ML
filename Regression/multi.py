import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(0)
X1 = 2*np.random.rand(100,1)
X2 = 3*np.random.rand(100,1)

y = 4 + 3 * X1 + 5 * X2 + np.random.randn(100, 1)

X = np.hstack([X1,X2])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model = LinearRegression()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

coefficients = model.coef_[0]
intercept = model.intercept_[0]

# Print the coefficients
print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Visualize the results (optional)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df.plot(kind='bar', figsize=(10, 6))
plt.title('Multiple Linear Regression')
plt.xlabel('Data points')
plt.ylabel('Values')
plt.show()