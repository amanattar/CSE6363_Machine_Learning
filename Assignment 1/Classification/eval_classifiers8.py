import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import LinearDiscriminantAnalysis

data = datasets.load_iris()
sepal_data = data.data[:, :2]
y = data.target

equal_split = y
X_train_sepal, X_test_sepal, y_train_sepal, y_test_sepal = train_test_split(sepal_data, y, test_size=0.1, random_state=6601, stratify=equal_split)

model = LinearDiscriminantAnalysis.LinearDiscriminantAnalysis()
model.fit(X_train_sepal, y_train_sepal)

y_pred_for_sepal = model.predict(X_test_sepal)
acc_for_sepal = accuracy_score(y_test_sepal, y_pred_for_sepal)
print("Accuracy for sepal data: " ,acc_for_sepal)

plot_decision_regions(X_train_sepal, y_train_sepal, clf=model, legend=2)
plt.xlabel('sepal Length')
plt.ylabel('sepal Width')
plt.title('Linear Discriminant Analysis (sepal Length, sepal Width)')
plt.show()