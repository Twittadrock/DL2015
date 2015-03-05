'''
Created on Mar 3, 2015

@author: Matthias
'''
import sys;
import theano
import numpy as np
import time
from theano import config


if __name__ == '__main__':
    print("%x" % sys.maxsize, sys.maxsize > 2**32)
    A = np.random.rand(1000,10000).astype(config.floatX)  # @UndefinedVariable
    B = np.random.rand(10000,1000).astype(config.floatX)  # @UndefinedVariable
    np_start = time.time()
    AB = A.dot(B)
    np_end = time.time()
    X,Y = theano.tensor.matrices('XY')
    mf = theano.function([X,Y],X.dot(Y))
    t_start = time.time()
    tAB = mf(A,B)
    t_end = time.time()
    print "NP time: %f[s], theano time: %f[s] (times should be close when run on CPU!)" %(
                                               np_end-np_start, t_end-t_start)
    print "Result difference: %f" % (np.abs(AB-tAB).max(), )
    
    
    pass