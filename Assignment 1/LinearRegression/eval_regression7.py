import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import LinearRegression


# load data
iris = datasets.load_iris()
model7 = LinearRegression.LinearRegression()
X_petal = iris.data[:,[2,3]]
y_sepal_length = iris.data[:,[0]]
equal_split = iris.target

model7.load('model7.pkl')


# split data
X_train, X_test, y_train, y_test = train_test_split(X_petal, y_sepal_length, test_size=0.1, random_state=6601, stratify=equal_split)
error = model7.score(X_test, y_test)

print(f"Mean Squared Error: {error}")



