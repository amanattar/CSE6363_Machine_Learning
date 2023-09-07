import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import LinearRegression


# load data
iris = datasets.load_iris()
model6 = LinearRegression.LinearRegression()
X_sepal = iris.data[:,[0,1]]
y_petal_width = iris.data[:,[3]]
equal_split = iris.target

model6.load('model6.pkl')


# split data
X_train, X_test, y_train, y_test = train_test_split(X_sepal, y_petal_width, test_size=0.1, random_state=6601, stratify=equal_split)
error = model6.score(X_test, y_test)

print(f"Mean Squared Error: {error}")



