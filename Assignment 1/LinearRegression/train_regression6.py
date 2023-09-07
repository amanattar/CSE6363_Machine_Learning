import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import LinearRegression


iris = datasets.load_iris()
model6 = LinearRegression.LinearRegression()

X_sepal = iris.data[:,[0,1]]
y_petal_width = iris.data[:,[3]]
equal_split = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_sepal, y_petal_width, test_size=0.1, random_state=6601, stratify=equal_split)

model6.fit(X_train, y_train, regularization= 0.2)

model6.save('model6.pkl')

print(f"Mean Squared Error: {model6.mean_sqrd_error}")


plt.title('Loss with steps, with regularization 0.2')

x11 = np.arange(len(model6.mse))
y11 = model6.mse

plt.plot(x11, y11, label='mse', color='red')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

