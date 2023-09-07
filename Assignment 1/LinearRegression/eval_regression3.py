import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import LinearRegression


# load data
iris = datasets.load_iris()
model3 = LinearRegression.LinearRegression()
X_petal_length = iris.data[:,[2]]
y_petal_width = iris.data[:,[3]]
equal_split = iris.target

model3.load('model3.pkl')


# split data
X_train, X_test, y_train, y_test = train_test_split(X_petal_length, y_petal_width, test_size=0.1, random_state=6601, stratify=equal_split)
error = model3.score(X_test, y_test)

print(f"Mean Squared Error: {error}")



