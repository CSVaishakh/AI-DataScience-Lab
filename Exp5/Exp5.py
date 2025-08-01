from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,precision_score
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt

data = load_breast_cancer()
X = data.data
Y = data.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.3, random_state=143)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
print(f"\nAccuracy : {accuracy}\nPrecision : {precision}\n")

plt.figure(figsize = (28,14))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, rounded=True,filled = True)
plt.title("Decision Tree")
plt.show()