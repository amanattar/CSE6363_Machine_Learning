import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import LinearDiscriminantAnalysis

data = datasets.load_iris()
petal_data = data.data[:, 2:4]
y = data.target

equal_split = y

X_train_petal, X_test_petal, y_train_petal, y_test_petal = train_test_split(petal_data, y, test_size=0.1, random_state=6601, stratify=equal_split)

model = LinearDiscriminantAnalysis.LinearDiscriminantAnalysis()
model.fit(X_train_petal, y_train_petal)

y_pred_for_petal = model.predict(X_test_petal)
acc_for_petal = accuracy_score(y_test_petal, y_pred_for_petal)
print("Accuracy for petal data: " ,acc_for_petal)

plot_decision_regions(X_train_petal, y_train_petal, clf=model, legend=2)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Linear Discriminant Analysis (Petal Length, Petal Width)')
plt.show()