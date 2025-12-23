import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
np.random.seed(42)


x = np.arange(0, 50).reshape(-1, 1)
y = 3 * x.flatten() + 5 + np.random.randn(50) * 10

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
print(model.coef_, model.intercept_)

y_pred = model.predict(x_test)

print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))


plt.scatter(x_test, y_test, color='blue', label='test data')
plt.scatter(x_train, y_train, color='green', label='training data')
x_line = np.linspace(0, 50, 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', label='Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


