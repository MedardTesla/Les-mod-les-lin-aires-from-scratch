import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))


class LogisticRegression():
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None
        
    
    def fit(self, x, y):
        n_sample, n_feature = x.shape
        y = y.reshape(n_sample, 1)
        X = np.hstack((x, np.ones((n_sample, 1))))
        self.theta = np.random.randn(n_feature+1, 1)
        
        for _ in range(0, self.n_iters):
            model_linear = X.dot(self.theta)
            model = sigmoid(model_linear)
            
            gradient = 1/n_sample * X.T.dot(model - y)
            self.theta = self.theta - self.lr*gradient
            
            
    
    def predict(self, x):
        n_sample, n_feature = x.shape
        X = np.hstack((x, np.ones((n_sample, 1))))
        y_pred_linear = X.dot(self.theta)
        y_pred = sigmoid(y_pred_linear)
        
        return [0 if y<=0.5 else 1  for y in y_pred]
        
        
        