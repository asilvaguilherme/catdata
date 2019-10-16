'''
Created on 12 oct. 2019

@author: galvesda
'''
import pandas as pd

def p2(X):
    
    n = len(X)
    den = n * (n - 1)
    hist = []    
    for c in X.T:
        fk = pd.Series(c).value_counts()
        hist.append((fk * (fk - 1)) / den)
        
    return hist

def f(X):
    
    n = len(X)
    hist = []    
    for c in X.T:
        fk = pd.Series(c).value_counts()
        hist.append(fk / n)
        
    return hist


def freq(X):
    
    hist = []    
    for c in X.T:
        fk = pd.Series(c).value_counts()
        hist.append(fk)
        
    return hist