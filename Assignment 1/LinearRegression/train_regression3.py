import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import LinearRegression


iris = datasets.load_iris()
model3 = LinearRegression.LinearRegression()

X_petal_length = iris.data[:,[2]]
y_petal_width = iris.data[:,[3]]
equal_split = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_petal_length, y_petal_width, test_size=0.1, random_state=6601, stratify=equal_split)

model3.fit(X_train, y_train)

model3.save('model3.pkl')

print(f"Mean Squared Error: {model3.mean_sqrd_error}")


plt.title('Loss with steps, without regularization')

x11 = np.arange(len(model3.mse))
y11 = model3.mse

plt.plot(x11, y11, label='mse', color='red')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

