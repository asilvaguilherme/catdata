'''
Created on 12 oct. 2019

@author: galvesda
'''

import multiprocessing as mp
import numpy as np


class Overlap():
    
    def __init__(self, X):
        self.X = X.values
        self.n_attrib = len(X.values[0])
        self.n = len(X)

    def diss_matrix(self):
         
        matrix = np.zeros(shape=(self.n,self.n))
        
        pool = mp.Pool(processes=mp.cpu_count())
        
        for i in range(self.n):
            pivot_vector = self.X[i,:]
            
            args = [(pivot_vector, self.X[j,:]) for j in range(i,self.n)]
            res = np.array(pool.starmap(self.overlap,args))
                
            matrix[i] = np.concatenate((np.zeros(shape=(i)),res))
            
                  
        for i in range(self.n):
            for j in range(i):
                matrix[i,j] = matrix[j,i]
        
        return matrix
    
    
    def overlap(self,X,Y):
        
        return 1 - ( np.sum(np.array([self.attib(X[i], Y[i]) for i in range(self.n_attrib)]) / self.n_attrib ))
        
        # Conversion according to page (from similarity to dissimilarity)
        # Sulc, Z., & Rezankovas, H. (2019). Comparison iof Similarity Measures for Categorical Data in Hierarchical Clustering. 
        # Journal iof Classification, 36(1), 58-72.
    
    def attib(self,X_k,Y_k):
        
        if X_k != Y_k:
            return 0
        else:
            return 1 
    