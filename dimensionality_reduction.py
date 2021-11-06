import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as library_pca



class PCA:
    # Just for the visualization ---> Not framed Yet
    """
    [Returns the n-components present in PCA]
    """    
    def __init__(self, n_components):
        self.n_components = n_components
    
    def mean(self, X):
        return sum(X)/len(X)

    def centerize(self, X):
        return X-np.mean(X, axis = 0)
    

    def get_cov(self, X):
        cov_matrix = np.cov(X)
        return cov_matrix
    
    def sort_eigenvalue(self, eigens):
        pass

    def transform(self, X):
        # print(X)
        X = self.centerize(X)
        feature_vector = (self.top_n_eigenvectors@X.T).T # ((n_components*m)*(m*n)).T
        return feature_vector

    def fit(self, X):
        X= self.centerize(X) # n*m
        covariance_matrix = self.get_cov(X.T) # m*m

        eigenvalue, eigenvector = np.linalg.eig(covariance_matrix)
        eigenvector = eigenvector.T
        sorted_eigens_args = np.argsort(-eigenvalue)
        eigens = list(zip(eigenvalue,eigenvector))
        sorted_eigens = sorted(eigens, key=lambda x: x[0], reverse = True)
        sorted_eigenvectors = [eigen[1] for eigen in sorted_eigens]
        top_n_eigenvectors = sorted_eigenvectors[:self.n_components]
        self.top_n_eigenvectors = top_n_eigenvectors






if __name__ == "__main__":

    X = np.array([[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]])
    y = np.array([[2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]])

    data = np.concatenate((X,y), axis = 0).T
    print("Original data --> \n", data)
    # print(data) #n*m, m = 2
    plt.show()
    pca = PCA(2)
    pca.fit(data)
    new_X = pca.transform(data)
    
    X = pca.centerize(data)
    plt.scatter(X[:,0], X[:,1], label='old_data', c='orange')
    plt.scatter(new_X[:,0], new_X[:,1], label='new_data', c='b')
    plt.legend()
    plt.savefig('Images/PCA_fit.png')
    plt.title("PCA")
    plt.show()
    print("\n\nTransformed Data without library--> \n",new_X, "\n\n")


    # With Library
    pca_2 = library_pca(n_components=2)
    new_X = pca_2.fit_transform(data)
    print("With library --> \n", new_X, "\n\n")
    plt.scatter(X[:, 0], X[:, 1], label = 'old_data', color='orange')
    plt.scatter(new_X[:, 0], new_X[:, 1], label = 'new_data', color='blue')
    plt.title("PCA (with_library)")
    plt.show()