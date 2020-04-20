import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class k_means:

    def __init__(self,clusters_no,maximum_iteration,centroids,last_centroids):
        self.no_of_clusters=clusters_no
        self.total_iter=maximum_iteration
        self.original_centroids=[]
        self.last_centroids=[]

    def calc_distances():
        
    def euclidean(x,y):
        return norm(x-y)



if __name__ == "__main__":
    dataset=pd.read_csv("durudataset.txt")
    num_instances,num_features=dataset.shape
    