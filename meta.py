import pickle
import random

import scipy
from sklearn import svm, metrics, preprocessing
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection._split import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier

from eskin import Eskin
from goodall import Goodall
from histogram import build_histogram, build_simple_fixed_bins_histogram
from iof import IOF
from lin import Lin
import numpy as np
from overlap import Overlap
import pandas as pd


def experiment(datasets):
    
    meta_X, meta_Y, Z, A = build_metadataset(datasets) # Z contains the best ARI achieved for each meta-instance
    
    outfile = open("metadaset-small2",'wb')
    pickle.dump([meta_X, meta_Y, Z, A],outfile)
    outfile.close()
    
    return 0


def build_metadataset(datasets):
    
    meta_dataset = []
    total = len(datasets)
    
    for X, Y, n_clusters in datasets:
        print("remaining",total)
        matrices = []
        
        matrices.append(Eskin(X).diss_matrix())
        
        matrices.append(Overlap(X).diss_matrix())
        
        lin = Lin(X)
        matrices.append(lin.diss_matrix_lin())
        matrices.append(lin.diss_matrix_lin1())
          
        iof = IOF(X)
        matrices.append(iof.diss_matrix_of())
        matrices.append(iof.diss_matrix_iof())
          
        goodall = Goodall(X)
        matrices.append(goodall.diss_matrix_g1())
        matrices.append(goodall.diss_matrix_g2())
        matrices.append(goodall.diss_matrix_g3())
        matrices.append(goodall.diss_matrix_g4())
        
        
        best_ari = None
        best_ari_index = None
        all_ari = []
        
        for i in range(len(matrices)):
            labels_pred = cluster(matrices[i], n_clusters)
            current_ari = metrics.adjusted_rand_score(Y[['class']].values.T[0], labels_pred)
            all_ari.append(current_ari)
            
            if (best_ari == None) or (current_ari > best_ari):
                best_ari = current_ari
                best_ari_index = i
        
        meta_fetaures = comp_meta_features(X)
        
        meta_dataset.append((meta_fetaures, best_ari_index, best_ari, all_ari))
        
        total -= 1
    
    print(meta_dataset)
    
    return np.array([x for x,_,_,_ in meta_dataset]), np.array([y for _,y,_,_ in meta_dataset]), np.array([z for _,_,z,_ in meta_dataset]), np.array([a for _,_,_,a in meta_dataset])

def comp_meta_features(dataset):
    
    metafeatures = []
        
    n_attrib = len(dataset.columns) # num of attributes
    n_instances = len(dataset) # num of instances
    dimensionality = n_attrib / n_instances# dimentionality
    
    metafeatures.append(n_attrib)
    metafeatures.append(n_instances)
    metafeatures.append(dimensionality)
    
    entropy_list = [] 
    skewness_list = []
    kurtosis_list = []
    for column in list(dataset):
        idx = pd.Index(dataset[column])
        p_data = idx.value_counts() # counts occurrence of each value
        entropy_list.append([scipy.stats.entropy(p_data)]) # get entropy from counts
        skewness_list.append([scipy.stats.skew(p_data)]) 
        kurtosis_list.append([scipy.stats.kurtosis(p_data)]) 
    
    scaled = preprocessing.MinMaxScaler().fit_transform(np.array(entropy_list))
    entropy_hist = build_simple_fixed_bins_histogram(scaled.T[0])
    metafeatures += entropy_hist
    
    scaled = preprocessing.MinMaxScaler().fit_transform(np.array(skewness_list))
    skewness_hist = build_simple_fixed_bins_histogram(scaled.T[0])
    metafeatures += skewness_hist
    
    scaled = preprocessing.MinMaxScaler().fit_transform(np.array(kurtosis_list))
    kurtosis_hist = build_simple_fixed_bins_histogram(scaled.T[0])
    metafeatures += kurtosis_hist

    return metafeatures

def cluster(diss_matrix, n_clusters):
    
    agglo_clusterer = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=n_clusters)
    return agglo_clusterer.fit(diss_matrix).labels_

def build_and_test_model(classifier, X, Y, Z, param):

    accuracies = []
    ari = []
    
    for train, test in LeaveOneOut().split(X):
        
        X_train, Y_train = X[train], Y[train]
        X_test, Y_test, Z_test = X[test], Y[test], Z[test]
        predicted = None
        
        if classifier == "KNN":
            neigh = KNeighborsClassifier(n_neighbors=param).fit(X_train, Y_train)
            predicted = neigh.predict(X_test)
            
        elif classifier == "RF":
            clf = RandomForestClassifier(n_estimators=param, random_state=0) # ,max_depth=2,
            clf.fit(X_train, Y_train)
            predicted = clf.predict(X_test)
        
        elif classifier == "SVM":
            clf = svm.SVC(gamma='scale')
            clf.fit(X_train, Y_train) 
            predicted = clf.predict(X_test).astype(int)
        
        elif classifier == "NAIVE":
            clf = GaussianNB()
            clf.fit(X_train, Y_train) 
            predicted = clf.predict(X_test).astype(int) 
        
        elif classifier == "RANDOM":
            options = list(set(Y_train))
            predicted = [random.choice(options) for _ in range(len(Y_test))]
            
        accuracies.append(metrics.accuracy_score(Y_test, predicted))
        ari.append(metrics.adjusted_rand_score(Z_test, predicted))
    
    return np.mean(accuracies), np.std(accuracies), np.mean(ari), np.std(ari)