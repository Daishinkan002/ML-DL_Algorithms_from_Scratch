import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt



class PCA:
    # Just for the visualization ---> Not completed Yet
    """
    [Returns the n-components present in PCA]
    """    
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
    
    def mean(self, X):
        return sum(X)/len(X)

    def centering(self, X, y):
        return X-self.mean(X), y-self.mean(y)
    
    def find_covariance(self, X,y):
        mean_X = self.mean(X)
        mean_y = self.mean(y)
        n = len(X)
        return sum([(X[i]-mean_X)*(y[i]-mean_y) for i in range(n)])/(n-1)

    def find_cov_matrix(self, X, y):
        cov_matrix = [[self.find_covariance(X,X), self.find_covariance(X,y)], [self.find_covariance(y,X), self.find_covariance(y,y)]]
        return cov_matrix
    
    def sort_eigenvalue(self, eigens):
        pass


    def fit(self, X, y):
        X,y = self.centering(X,y)
        covariance_matrix = self.find_cov_matrix(X,y)
        eigenvalue, eigenvector = eig(covariance_matrix)
        eigens = list(zip(eigenvalue,eigenvector))
        sorted_eigens = sorted(eigens, key=lambda x: x[0])
        print(sorted_eigens)
        X = np.array(list(zip(X,y)))
        line_1 = [[0, sorted_eigens[0][1][0]], [0, sorted_eigens[0][1][1]]]
        line_2 = [[0, sorted_eigens[1][1][0]], [0, sorted_eigens[1][1][1]]]
        plt.axline((0, 0), (sorted_eigens[0][1][0], sorted_eigens[0][1][1]))
        feature_vector = (np.array(-eigenvector[0]).T)@X.T
        plt.scatter(feature_vector, y)
        plt.show()






if __name__ == "__main__":

    X = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1])
    y = np.array([2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9])

    model = PCA(1)
    model.fit(X,y)
    plt.scatter(X,y)
    plt.show()