import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import LogisticRegression

data = datasets.load_iris()
all_data = data.data
y = data.target

equal_split = y
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(all_data, y, test_size=0.1, random_state=6601, stratify=equal_split)

model = LogisticRegression.LogisticRegression()
model.fit(X_train_all, y_train_all,learning_rate=0.3)

y_pred_for_all = model.predict(X_test_all)
acc_for_all = accuracy_score(y_test_all, y_pred_for_all)
print("Accuracy for all data: " ,acc_for_all)

