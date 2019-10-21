'''
Created on 12 oct. 2019

@author: galvesda
'''

import math

import multiprocessing as mp
import numpy as np
import stats


class IOF():
    
    def __init__(self, X):
        self.X = X.values
        self.n_attrib = len(X.values[0])
        self.n = len(X)
        self.f = stats.freq(X.values)
        self.cache = np.array([dict() for _ in range(self.n_attrib)])

    def diss_matrix_of(self):
         
        matrix = np.zeros(shape=(self.n,self.n))
        
        pool = mp.Pool(processes=mp.cpu_count()-1)
        
        for i in range(self.n):
            pivot_vector = self.X[i,:]
            
            args = [(pivot_vector, self.X[j,:], self.f) for j in range(i,self.n)]
            res = np.array(pool.starmap(self.of,args))
                
            matrix[i] = np.concatenate((np.zeros(shape=(i)),res))
            
                  
        for i in range(self.n):
            for j in range(i):
                matrix[i,j] = matrix[j,i]
        
        return matrix
    
    def diss_matrix_iof(self):
         
        matrix = np.zeros(shape=(self.n,self.n))
        
        pool = mp.Pool(processes=mp.cpu_count()-1)
        
        for i in range(self.n):
            pivot_vector = self.X[i,:]
            
            args = [(pivot_vector, self.X[j,:], self.f) for j in range(i,self.n)]
            res = np.array(pool.starmap(self.iof,args))
                
            matrix[i] = np.concatenate((np.zeros(shape=(i)),res))
            
                  
        for i in range(self.n):
            for j in range(i):
                matrix[i,j] = matrix[j,i]
        
        return matrix
    
    
    def of(self,X,Y,prob):
        return ( 1 / ( np.sum(np.array([self.attib_of(i, X[i], Y[i], prob) for i in range(self.n_attrib)]) / self.n_attrib )) ) - 1
        

    def iof(self,X,Y,prob):
        return ( 1 / ( np.sum(np.array([self.attib_iof(i, X[i], Y[i], prob) for i in range(self.n_attrib)]) / self.n_attrib )) ) - 1

    
    def attib_of(self,k,X_k,Y_k,freq):
        
        if X_k != Y_k:
            return 1 / ( 1 + math.log(self.n/freq[k][X_k]) * math.log(self.n/freq[k][Y_k]) )
        else:
            return 1
    
    def attib_iof(self,k,X_k,Y_k,freq):
        
        if X_k != Y_k:
            return 1 / (1 + math.log(freq[k][X_k]) * math.log(freq[k][Y_k]) )
        else:
            return 1