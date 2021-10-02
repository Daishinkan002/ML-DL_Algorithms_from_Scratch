import pandas as pd
import numpy as np
import matplotlib as plt
import random
import math
import sys
from sklearn.model_selection import train_test_split




class KNN(object):

    def __init__(self, k=3, algorithm='auto'):
        self.k = k
        self.X_train = []
        self.y_train = []
        self.algorithm=algorithm
    
    def euclidean_distance(self,a,b):
        dist = 0
        length = len(b)
        for i in range(length):
            dist += (a[i]-b[i])**2
        res = math.sqrt(dist)
        return res

    def cosine_similarity_distance(self, xi, xq):
        length_xi = len(xi)
        mag_xi_square = 0
        for j in range(length_xi):
            mag_xi_square += xi[j]
        mag_xi = math.sqrt(mag_xi_square)
        xi = np.array(xi)/mag_xi


        length_xq = len(xq)
        mag_xq_square = 0
        for j in range(length_xq):
            mag_xq_square += xq[j]
        mag_xq = math.sqrt(mag_xq_square)
        xq = np.array(xq)/mag_xq

        return 1-(xi @ xq.T)

    
    def fit(self, X, y):
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()
    
    def predict(self, X_test):
        X_test = X_test.to_numpy()
        length = len(X_test)
        y_pred = []
        for i in range(length):
           y_pred.append(self.predict_single(X_test[i]))
        return y_pred

    def predict_single(self, x):
        if(self.algorithm=='auto'):
            distance_list = []
            for i in range(len(self.X_train)):
                dist = self.euclidean_distance(x, self.X_train[i])
                distance_list.append([self.y_train[i],dist])
            distance_list.sort(key = lambda x:x[1])
            neighbours = distance_list[0:self.k]
            output = self.majority_vote(neighbours)
            return output


    def majority_vote(self, neighbours):
        a = []
        dictionary = {}
        for i in range(self.k):
            label = neighbours[i][0]
            try:
                dictionary[label] += 1
            except:
                dictionary[label] = 1
        
        max_value = 0
        max_occuring_label = []
        output = ''
        for key, value in dictionary.items():
            if(value > max_value):
                output = key
                max_value = value
        return output

    



    def score(self, X,y):
        correct = 0
        length = len(X)
        y = y.to_numpy()
        y_pred = self.predict(X)
        for i in range(length):
            if(y_pred[i] == y[i]):
                correct +=1
        return 100*correct/float(length)


if __name__ == "__main__":
    df = pd.read_csv('Datasets/Iris.csv')
    X = df.drop(['Id', 'Species'], axis=1)
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    k = 5
    model = KNN(k)
    model.fit(X_train, y_train)
    print(f"Model Accuracy (when k = {k}) is {model.score(X_test, y_test)}")

