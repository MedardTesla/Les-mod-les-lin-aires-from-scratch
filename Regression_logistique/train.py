import numpy as np
from regression_logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(lr=0.01, n_iters=1000)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)


def accuracy(y_pred, y):
    return np.sum(y_pred==y)/len(y)

print("Accuracy", accuracy(prediction, y_test))