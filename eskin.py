'''
Created on 12 oct. 2019

@author: galvesda
'''

import multiprocessing as mp
import numpy as np
import stats


class Eskin():
    
    def __init__(self, X):
        self.X = X.values
        self.n_attrib = len(X.values[0])
        self.n = len(X)
        self.cache = np.array([dict() for _ in range(self.n_attrib)])
        self.f = stats.p2(X.values) # no matter if it is f or p2, the function only need the size of the domain for each attribute

    def diss_matrix(self):
         
        matrix = np.zeros(shape=(self.n,self.n))
        
        pool = mp.Pool(processes=mp.cpu_count())
        
        for i in range(self.n):
            pivot_vector = self.X[i,:]
            
            args = [(pivot_vector, self.X[j,:], self.f) for j in range(i,self.n)]
            res = np.array(pool.starmap(self.eskin,args))
            matrix[i] = np.concatenate((np.zeros(shape=(i)),res))
                  
        for i in range(self.n):
            for j in range(i):
                matrix[i,j] = matrix[j,i]
        
        return matrix
    
    
    def eskin(self,X,Y,hist):
        
        return 1 - ( np.sum(np.array([self.attrib_eskin(i, X[i], Y[i], hist) for i in range(self.n_attrib)]) / self.n_attrib )) # from similarity to dissimilarity
        # Conversion according to page 
        # Sulc, Z., & Rezankovas, H. (2019). Comparison of Similarity Measures for Categorical Data in Hierarchical Clustering. 
        # Journal of Classification, 36(1), 58-72.
    
    def attrib_eskin(self,k,X_k,Y_k,hist):
        
        nk2 = len(hist[k]) ** 2
        
        if X_k != Y_k:
            return 0
        
        return ( nk2 / (nk2 + 2) )