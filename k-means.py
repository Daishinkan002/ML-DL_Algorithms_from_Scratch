import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt

infinity = 9999

class k_means:
    def __init__(self,clusters_no,clusters,iterations=1000):
        self.no_of_clusters=clusters_no
        self.clusters={}
        self.iterations=iterations
        for i in range(self.no_of_clusters):
            self.clusters[i]=[]

    def init_clusters(self):
        for i in range(self.no_of_clusters):
            self.clusters[i]=[]

    def init_centroids(self,features):
        centroids = []
        for i in range(self.no_of_clusters):
            centroids.append(features[np.random.randint(0,197)])
        return centroids

    def clear_clusters(self):
        for i in range(self.no_of_clusters):
            self.clusters[i]=[]

    def euclidean(self,x,y):
        return norm(x-y)

    def assigning_cluster(self,features,centroid):
        for i in range(len(features)):
            assignment = 0
            distance = infinity
            for j in range(self.no_of_clusters):
                calc_distance = self.euclidean(np.array(features[i]),np.array(centroid[j]))
                if(distance > calc_distance):
                    assignment = j
                    distance = calc_distance
            self.clusters[assignment].append(features[i])


    def calc_new_centroids(self,features,centroids):
        self.clear_clusters()
        self.assigning_cluster(features,centroids)
        for i in range(self.no_of_clusters):
            centroids[i]=np.mean(self.clusters[i])
        return centroids


    def fit(self,features):
        original_centroids = self.init_centroids(features)
        history_centroids = np.zeros((self.no_of_clusters,), dtype=[('x', 'int'), ('y', 'int')])
        counter=0
        while(counter < self.iterations):
            if(np.array_equal(history_centroids,original_centroids)):
                break
            history_centroids=original_centroids.copy()
            original_centroids=self.calc_new_centroids(features,original_centroids)
        return_cluster = {}

        for i in range(self.no_of_clusters):
            return_cluster[i]=self.clusters[i]
        return return_cluster 


if __name__ == "__main__":
    dataset=pd.read_csv("dataset.csv")
    features = dataset[['Annual Income (k$)','Spending Score (1-100)']]
    plt.scatter(features['Annual Income (k$)'],features['Spending Score (1-100)'])
    plt.show()
    no_of_cluster=int(input("Enter number of clusters : "))
    clusters={}
    model = k_means(no_of_cluster,clusters,1000)
    data=features.values.tolist()
    clusters=model.fit(data)
    color_dict = {'1': "blue", '2': "orange", '3': "green", '4': "yellow", '5' : "purple", '6' : "red", '7' : "black"}
    for i in range(len(clusters)):
        j = np.array(clusters[i])
        plt.scatter(j[:,0], j[:,1], c = color_dict[str(i+1)])
    plt.show()
