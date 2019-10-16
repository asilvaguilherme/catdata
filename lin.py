'''
Created on 12 oct. 2019

@author: galvesda
'''

import math

import multiprocessing as mp
import numpy as np
import stats


class Lin():
    
    def __init__(self, X):
        self.X = X.values
        self.n_attrib = len(X.values[0])
        self.n = len(X)
        self.f = stats.freq(X.values)
        self.cache = np.array([dict() for _ in range(self.n_attrib)])

    def diss_matrix(self, f="lin"):
         
        matrix = np.zeros(shape=(self.n,self.n))
        
        pool = mp.Pool(processes=mp.cpu_count())
        
        for i in range(self.n):
            pivot_vector = self.X[i,:]
            
            args = [(pivot_vector, self.X[j,:], self.f, f) for j in range(i,self.n)]
            res = np.array(pool.starmap(self.iof,args))
                
            matrix[i] = np.concatenate((np.zeros(shape=(i)),res))
            
                  
        for i in range(self.n):
            for j in range(i):
                matrix[i,j] = matrix[j,i]
        
        return matrix
    
    
    def iof(self,X,Y,prob,f):
        
        acm = np.sum(np.array([math.log(prob[i][X[i]]) for i in range(self.n_attrib)]))
        acm += np.sum(np.array([math.log(prob[i][Y[i]]) for i in range(self.n_attrib)]))

        if f == "lin":
            return 1 - ( np.sum(np.array([self.attib_lin(i, X[i], Y[i], prob) for i in range(self.n_attrib)]) / acm ))
        elif f == "lin1":
            return 1 - ( np.sum(np.array([self.attib_lin1(i, X[i], Y[i], prob) for i in range(self.n_attrib)]) / acm )) 
        
        # Conversion according to page (from similarity to dissimilarity)
        # Sulc, Z., & Rezankovas, H. (2019). Comparison iof Similarity Measures for Categorical Data in Hierarchical Clustering. 
        # Journal iof Classification, 36(1), 58-72.
    
    def attib_lin(self,k,X_k,Y_k,freq):
        
        if X_k != Y_k:
            return 2 * math.log(freq[k][X_k] + freq[k][Y_k])
        else:
            return 2 * math.log(freq[k][X_k]) 
    
    def attib_lin1(self,k,X_k,Y_k,prob):
        
        acm = None
        
        if X_k in self.cache[k]:
            acm = self.cache[k][X_k]
        else:
            sub_hist = prob[k]
            pvt_min = sub_hist[X_k] # pk(X_k)
            pvt_max = sub_hist[Y_k] # pk(X_k)
            
            if pvt_max < pvt_min:
                pvt_min, pvt_max = pvt_max, pvt_min
            
            acm = np.sum(np.array([value for value in sub_hist if pvt_min <= value and value <= pvt_max]))
            
            self.cache[k][X_k] = acm
            
        if X_k != Y_k:
            return 2 * math.log(acm)
        else:
            return acm
        
        return 0