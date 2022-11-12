import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from regression_lineaire import RegressionLineaire


X, y = datasets.make_regression(n_samples=200, n_features=1, noise=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RegressionLineaire(lr=0.01, n_iters=1000)
model.fit(X, y)



def mse(y, y_pref):
    m = len(y)
    return (1/2*m)*np.sum((y_pref-y)**2)




def coef_determination(y, prediction):
  y = y.reshape(y.shape[0], 1)
  u = ((y - prediction)**2).sum()
  v = ((y - y.mean())**2).sum()
  
  return 1 - u/v


print("Le coefficient de determination est : ", coef_determination(y_train, model.predict(X_train)))