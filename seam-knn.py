import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics

seam = pd.read_csv('sample-torso-6.csv', encoding="utf-8", header=0)

target = seam.columns[-1]
X = seam.drop(target, axis=1)
y = seam.drop(X, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=1)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

prediction_accuracy = []
k_value = []

# Calculating prediction accuracy for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train.values.ravel())
    pred_i = knn.predict(X_test)
    prediction_accuracy.append(metrics.accuracy_score(y_test, pred_i))
    k_value.append(i)

# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 40), prediction_accuracy, color='red', linestyle='dashed', marker='o',
#          markerfacecolor='blue', markersize=10)
# plt.title('Prediction Accuracy vs K Value')
# plt.xlabel('K Value')
# plt.ylabel('Prediction Accuracy')
# plt.show()

doof = sorted(zip(prediction_accuracy, k_value), reverse=True)[:3]

print(sorted(zip(prediction_accuracy, k_value), reverse=True)[:3])

bestest_k_dudes = [item[1] for item in doof]

for i in bestest_k_dudes:
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train.values.ravel())

    y_pred = classifier.predict(X_test)

    print(classification_report(y_test, y_pred))
