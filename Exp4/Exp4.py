from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix,precision_score,f1_score

irisData = load_iris()

X = irisData.data
Y = irisData.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.2, random_state=453)
knn = KNeighborsClassifier(n_neighbors = 3).fit(x_train,y_train)

pred = knn.predict(x_test)

accuracy = knn.score(x_test,y_test)
con_mat = confusion_matrix(y_test,pred)
f1 = f1_score(y_test,pred, average="macro")
precision = precision_score(y_test,pred, average="macro")

correct_predictions = 0
wrong_predictions = 0

print("\nCORRECT PREDICTIONS:")
print("-" * 30)
for i in range(len(y_test)):
    if pred[i] == y_test[i]:
        correct_predictions += 1
        print(f"Sample {i+1}: Predicted = {irisData.target_names[pred[i]]}, "
              f"Actual = {irisData.target_names[y_test[i]]} ✓")

print(f"\nTotal Correct Predictions: {correct_predictions}")

print("\nWRONG PREDICTIONS:")
print("-" * 30)
for i in range(len(y_test)):
    if pred[i] != y_test[i]:
        wrong_predictions += 1
        print(f"Sample {i+1}: Predicted = {irisData.target_names[pred[i]]}, "
              f"Actual = {irisData.target_names[y_test[i]]} ✗")

print(f"\nTotal Wrong Predictions: {wrong_predictions}")

print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Total Test Samples: {len(y_test)}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Wrong Predictions: {wrong_predictions}")
print(f"Accuracy: {accuracy:.2f} = ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.2f}")
print(f"f1 score : {f1:.2f}")
print(f"Confusion Matrix : \n {con_mat}")