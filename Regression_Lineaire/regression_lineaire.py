import numpy as np


class RegressionLineaire():
    
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr                                    # le learning rate
        self.n_iters = n_iters                          # le nombre d'itération
        self.theta = None                               # Vecteur contenant le poids et le bias (a, b)
        
    
    def fit(self, x, y):
        
        n_samples, n_feature = x.shape
        y = y.reshape(y.shape[0], 1)
        X = np.hstack((x, np.ones((n_samples, n_feature))))
        self.theta = np.random.randn(2, 1)              # initialisation du theta avec des valeur aleatoire
        
       
        
        for i in range(0, self.n_iters):
            model = X.dot(self.theta)                      # (n_samples*n_feature+1)*(2, 1)
            gradient = 1/n_samples * X.T.dot(model - y)     # dimention  2*1
            self.theta = self.theta - self.lr * gradient
            
   
    def predict(self, x):
        n_samples, n_feature = x.shape
        X = np.hstack((x, np.ones((n_samples, n_feature))))
        return X.dot(self.theta)
    
