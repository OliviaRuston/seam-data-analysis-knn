import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection  import cross_val_score

torso = pd.read_csv('torso_design2_6.csv', encoding="utf-8", header=0)

target = torso.columns[-1]
X = torso.drop(target, axis=1)
y = torso.drop(X, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=1)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(X_train, y_train.values.ravel())
#
# y_pred = classifier.predict(X_test)
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train.values.ravel())
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test.values.ravel()))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()