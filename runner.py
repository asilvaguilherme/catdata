import random
import sys

from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

from iof import IOF
import meta
import numpy as np
import pandas as pd


# from kmodes.kmodes import KModes
# data = np.random.choice(20, (100, 10)) # random categorical data
# km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)
# clusters = km.fit_predict(data)
# print(km.cluster_centroids_) # Print the cluster centroids
################
# from sklearn.cluster import KMeans
# X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# kmeans.predict([[0, 0], [12, 3]])
path = "C:\\Users\\galvesda\\Desktop\\categorical datasets\\" # 1:4
# path = "C:\\Users\\galvesda\\Desktop\\categorical datasets\\" # 0:35
# path = "C:\\Users\\galvesda\\Desktop\\categorical datasets\\mushrooms\\mushrooms.data" # 1:22
sep = ","
n_clusters = 3

def run():
    
    datasets = []
    
    X1, Y1 = load_dataset(path + "lenses\\lenses.data", range(1,4), sep)
    datasets.append((X1,Y1,3))
    
    X2, Y2 = load_dataset(path + "soybean\\soybean-small.data", range(0,35), sep)
    datasets.append((X2,Y2,4))
    
    meta.experiment(datasets)
    
#     distances = IOF(X).diss_matrix(f="iof")
#      
#     agglo_clusterer = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=n_clusters)
#     current_partition = agglo_clusterer.fit(distances).labels_
#     
#     np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision=2, suppress=True)
# #     print(distances)
#     
#     print(current_partition)
#     print(Y[['class']].T.values)
#     
#     print(metrics.adjusted_rand_score(Y[['class']].T.values[0], current_partition))

 
def convert2(sim):
    return ( 1 / sim ) - 1 

def load_dataset(path, interval, sep):
    data = load_from_file(path,sep)
    
    indices = [i for i in range(1,len(data))]
    random.shuffle(indices)
    
    data = np.array(data)[[0]+indices]
    
    df = pd.DataFrame(data=data[1:],columns=data[0]) 
    
    X = df.iloc[:,interval]
    Y = df.loc[:,['class']]
    
    return X, Y

def load_from_file(path,separator):
    
    instances = []
    file = open(path,"r+",encoding='utf-8')
    
    for line in file:
        instances.append(line.strip('\n').split(separator))
    
    return instances

if __name__ == '__main__':
    run()