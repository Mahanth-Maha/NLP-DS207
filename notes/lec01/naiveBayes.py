import numpy as np

class NaiveBayes:
    def fit(self, X, y, alpha = 0):
        self.n,self.m = X.shape
        self.prior = np.zeros(len(np.unique(y)))
        self.likelihood = np.zeros((len(np.unique(y)), self.m))
        self.classes = np.unique(y)
        self.alpha = alpha

        for i in range(len(self.classes)):
            self.prior[i] = (np.sum(y==self.classes[i])+self.alpha)/(self.n+self.alpha*len(self.classes))
            for j in range(self.m):
                numm = np.sum(X[y==self.classes[i],j] > 0)
                denn = np.sum(y==self.classes[i])
                self.likelihood[i,j] = (numm+self.alpha)/(denn+self.alpha*len(self.classes))

        return np.log(self.prior), np.log(self.likelihood)

    def predict(self, X):
        self.pred = np.zeros((X.shape[0], len(self.classes)))
        for i in range(len(self.classes)):
            self.pred[:,i] = np.log(self.prior[i]) + np.sum(np.log(self.likelihood[i,:])*X, axis=1)
        return self.classes[np.argmax(self.pred, axis=1)]

    def evaluate(self, X, y):
        yp = self.predict(X)
        return self.acc_score(yp, y) 
    
    def acc_score(self,y_pred, y_test):
        return np.mean(y_pred == y_test)

