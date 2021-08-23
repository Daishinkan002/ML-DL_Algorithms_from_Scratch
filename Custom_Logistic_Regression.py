import numpy as np
import pandas as pd
import math
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm





class LogisticRegression:
    
    def __init__(self):
        self.lr = 0.1
        self.epochs = 100
        self.weights = []
    
    def calc_yhat(self, weights, x_train):
        exponential_terms = x_train@weights.T # [22*2] (*) [2*1] ----> [22*1]
        y_hat = []
        for i in range(len(exponential_terms)):
            exp = exponential_terms[i]
            y_hat.append(1/(1+math.pow(math.e, -exp))) #Needed [22*1]
        return np.array([y_hat]).T
        
    def fit(self, X_train, y_train):
        x_train = X_train.copy()
        x_train['const'] = 1
        x_train = np.array(x_train)
        self.weights = np.zeros((1, x_train.shape[1])) #1*2
        for i in range(self.epochs):
            y_pred = self.calc_yhat(self.weights, x_train) #(22,1)
            dw = self.gradient_descent(y_pred, y_train, x_train)
            self.weights += self.lr*dw
        
    def gradient_descent(self, y_pred, y_actual, x_train):
        difference = y_actual-y_pred # (22,1)
        dw = np.array(difference.T@x_train) #(1,22)*(22,2) ----> [1*2]
        return dw
    
    def get_weights(self):
        return self.weights

    def get_lr(self):
        return self.lr
    
    def predict(self, X_test):
        x_test = X_test.copy()
        x_test['const'] = 1
        exponential_terms = np.array(x_test@self.weights.T) # [8*2] (*) [2*1] ----> [22*1]
        y_hat = []
        for i in range(len(exponential_terms)):
            exp = exponential_terms[i]
            y_hat.append(1/(1+math.pow(math.e, -exp))) #Needed [22*1]
        y_hat = np.array([y_hat]).T
        return y_hat




x = [i for i in range(30)]
y = [0 if i<16 else 1 for i in range(30)]

df = pd.DataFrame({'x': x, 'y': y})
y = df[['y']]
X = df[['x']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, shuffle=True)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_test = np.array(y_test)

x = pd.DataFrame(np.linspace(1,30,100))
y = model.predict(x)

plt.plot(np.array(x), y)
plt.scatter(np.array(X_train), np.array(y_train), c='coral')
plt.show()