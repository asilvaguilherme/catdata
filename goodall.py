'''
Created on 12 oct. 2019

@author: galvesda
'''

import multiprocessing as mp
import numpy as np
import stats


class Goodall():
    
    def __init__(self, X):
        self.X = X.values
        self.n_attrib = len(X.values[0])
        self.n = len(X)
        self.cache = np.array([dict() for _ in range(self.n_attrib)])
        self.f = stats.p2(X.values)

    def diss_matrix(self, f="goodall1"):
         
        matrix = np.zeros(shape=(self.n,self.n))
        
        pool = mp.Pool(processes=mp.cpu_count())
        
        for i in range(self.n):
            pivot_vector = self.X[i,:]
            
            args = [(pivot_vector, self.X[j,:], self.f, f) for j in range(i,self.n)]
            res = np.array(pool.starmap(self.goodall,args))
                
            matrix[i] = np.concatenate((np.zeros(shape=(i)),res))
            
                  
        for i in range(self.n):
            for j in range(i):
                matrix[i,j] = matrix[j,i]
        
        return matrix
    
    
    def goodall(self,X,Y,freq,f):
        
        if f == "goodall1":
            return 1 - ( np.sum(np.array([self.attib_goodall1(i, X[i], Y[i], freq) for i in range(self.n_attrib)]) / self.n_attrib ))
        elif f == "goodall2":
            return 1 - ( np.sum(np.array([self.attib_goodall2(i, X[i], Y[i], freq) for i in range(self.n_attrib)]) / self.n_attrib )) 
        elif f == "goodall3":
            return 1 - ( np.sum(np.array([self.attib_goodall3(i, X[i], Y[i], freq) for i in range(self.n_attrib)]) / self.n_attrib ))
        elif f == "goodall4":
            return 1 - ( np.sum(np.array([self.attib_goodall4(i, X[i], Y[i], freq) for i in range(self.n_attrib)]) / self.n_attrib )) 
        
        # from similarity to dissimilarity
        # Conversion according to page 
        # Sulc, Z., & Rezankovas, H. (2019). Comparison of Similarity Measures for Categorical Data in Hierarchical Clustering. 
        # Journal of Classification, 36(1), 58-72.
    
    def attib_goodall1(self,k,X_k,Y_k,freq):
        
        if X_k != Y_k:
            return 0
        
        if X_k in self.cache[k]:
            return self.cache[k][X_k]
        
        sub_hist = freq[k]
        pivot = sub_hist[X_k] # pk(X_k)
        
        res = 1 - np.sum(np.array([value for value in sub_hist if value <= pivot]))
        
        self.cache[k][X_k] = res
        
        return res
    
    def attib_goodall2(self,k,X_k,Y_k,prob):
        
        if X_k != Y_k:
            return 0
        
        if X_k in self.cache[k]:
            return self.cache[k][X_k]
        
        sub_hist = prob[k]
        pivot = sub_hist[X_k] # pk(X_k)
        
        res = 1 - np.sum(np.array([value for value in sub_hist if value >= pivot]))
        
        self.cache[k][X_k] = res
        
        return res
    
    def attib_goodall3(self,k,X_k,Y_k,hist):
        
        if X_k != Y_k:
            return 0
        
        return 1 - hist[k][X_k] # pk(X_k)
    
    def attib_goodall4(self,k,X_k,Y_k,hist):
        
        if X_k != Y_k:
            return 0
        
        return hist[k][X_k] # pk(X_k)
    
