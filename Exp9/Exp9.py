from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
X, Y = iris.data, iris.target

x_train, x_test, y_train,y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:\n", accuracy)

importance = model.coef_

print("\nFeature importance (coefficients for each class):")
for i in range(len(importance)):
    print(f"Class {i}: {importance[i]}")