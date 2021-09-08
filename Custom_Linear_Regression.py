import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class LinearRegression(object):

    def __init__(self, learning_rate=0.001, epoch=1000):
        self.w = None
        self.lr = learning_rate
        self.epoch = epoch
        self.error = []
        self.epoch_cycle = []
        self.r2_score = None

    def fit(self, X, y, ridge_parameter = 0):        
        self.w = np.zeros((1, X.shape[1]))
        X = np.array(X)
        for i in range(self.epoch):
            y_pred = X.dot(self.w.T)
            self.error.append(self.__mse__(y, y_pred))
            self.epoch_cycle.append(i)
            dw = self.gradient_descent(y_pred, y, X)
            correction = -(self.lr * dw) + 2*ridge_parameter*self.w
            self.w += correction
        print(self.w)
    
    def gradient_descent(self, y_pred, y_act, X):
        y_act = np.array(y_act).reshape(len(y_act), 1)
        diff = y_act - y_pred
        dw = -2*diff.T@(X)
        return dw
    
    def predict(self, X_test):
        return X_test.dot(self.w.T)
    
    def __mse__(self, y, y_hat):
        y = np.array(y).reshape(len(y), 1)
        diff = y - y_hat
        return 0.5 * np.array(diff.T@diff)[0][0]
    
    def plot_error(self):
        plt.plot(self.epoch_cycle, self.error)
        plt.xlabel("epochs")
        plt.ylabel("MSE")
    
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)[0]
        y_test_pred = np.array(y_pred.values.tolist())
        y_test = np.array(y_test.values.tolist())
        ssr = np.sum((y_pred - y_test)**2)
        sst = np.sum((y_test - np.mean(y_test))**2)
        self.r2_score = 1 - (ssr/sst)
        return self.r2_score



# secondary_df = pd.read_csv('all_features.csv')
# X_one_degree = secondary_df[['theta', 'theta_dot', 'constant']]
# Y_one_degree = secondary_df['theta_ddot']
# from sklearn.model_selection import train_test_split

# X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_one_degree, Y_one_degree, test_size=0.2, random_state=42)
# lr_one_deg = LinearRegression()
# lr_one_deg.fit(X_1_train, y_1_train)
# lr_one_deg.plot_error()
# plt.show()