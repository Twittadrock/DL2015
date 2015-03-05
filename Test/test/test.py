'''
Created on Feb 27, 2015

@author: Matthias
'''

import os
import numpy as np
import matplotlib.pyplot as plt


def f(x,y):
    return x * y

def g(x):
    return x * x

if __name__ == '__main__':
    print os.listdir(os.getcwd())
    abc = []
    C = np.matrix([[1, 2], [3, 4]])
    print C
    B = C.reshape([2,2])
    print C.dot(B)
    print f(3,4)
    
    X = np.arange(1,10,1)
    print X
    Y = g(X)
    print Y
    
    plt.plot()
    plt.show()
    
    pass