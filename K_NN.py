import pandas as pd
import numpy as np
import matplotlib as plt
import random
import math




def load_dataset(dataset,split,training_set,testing_set):
    for x in range(len(dataset)-1):
        for y in range(len(dataset[0])-1):
            dataset[x][y] = float(dataset[x][y])
        if random.random() < split:
            training_set.append(dataset[x])
        else:
            testing_set.append(dataset[x])



def euclidean_distance(a,b,length):
    dist = 0
    for i in range(length):
        dist += (a[i]-b[i])**2
    res = math.sqrt(dist)
    #print(res)
    return res

def predict_neighbour(training_set,prediction_data, k):
    distance_list = []
    length = len(prediction_data)-1
    for i in range(len(training_set)):
        dist = euclidean_distance(prediction_data,training_set[i],length)
        distance_list.append([training_set[i],dist])
    distance_list.sort(key = lambda x:x[1])
    #print(distance_list,"\n\n")
    neighbours = distance_list[0:k]
    a = []
    for i in range(k):
        a.append(distance_list[i][0])
    #print(a)
    return a

def vote(point_list):
    vote_list = {}
    for i in range(len(point_list)):
        vote = point_list[i][-1]
        if vote in vote_list:
            vote_list[vote]+=1
        else:
            vote_list[vote]=1

           
    sorted_vote = sorted(vote_list.items(),key=lambda x:x[1], reverse=True)
    #print(sorted_vote)
    #print(vote_list,sorted_vote)
    #print("Sorted_Vote = ",sorted_vote[0])
    return(sorted_vote[0])



def find_acc(testing_set,inferences):
    correct = 0
    length = len(testing_set)
    for i in range(length):
        print(testing_set[i][-1],inferences[i])
        if testing_set[i][-1] == inferences[i]:
            correct +=1
    return 100*correct/float(length)


if __name__ == "__main__":
    dataset = pd.read_csv("Iris.csv")
    #features = dataset[['RI','Na','Mg','Al','Si','Type']]
    data=dataset.values.tolist()
    training_set = []
    testing_set = []
    validate = []
    load_dataset(data,0.67,training_set,testing_set)
    k = 6   
    #print(testing_set)
    for i in range(len(testing_set)):
        closest = predict_neighbour(training_set,testing_set[i],k)
        print("Closest = ",closest,"\ntraining set [i] = ",testing_set[i],"\n\n\n")
        res  = vote(closest)
        #print("RES = ",res,"\n\n\n")
        validate.append(res[0])
        #print(res[1],' ====== ',testing_set[i][-1] )
    accuracy = find_acc(testing_set,validate)
    print("Accuracy = ",accuracy)
