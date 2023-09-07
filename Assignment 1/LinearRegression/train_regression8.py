import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import LinearRegression


iris = datasets.load_iris()
model8 = LinearRegression.LinearRegression()

X_petal = iris.data[:,[2,3]]
y_sepal_length = iris.data[:,[0]]
equal_split = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_petal, y_sepal_length, test_size=0.1, random_state=6601, stratify=equal_split)

model8.fit(X_train, y_train, regularization= 0.2)

model8.save('model8.pkl')

print(f"Mean Squared Error: {model8.mean_sqrd_error}")


plt.title('Loss with steps, with regularization 0.2')

x11 = np.arange(len(model8.mse))
y11 = model8.mse

plt.plot(x11, y11, label='mse', color='red')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

