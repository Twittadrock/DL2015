'''
Created on Mar 4, 2015

@author: Matthias
'''

import numpy as np
from scipy.special import expit

class Hiddenlayer(object):
    '''
    classdocs
    '''


    def __init__(self, n_in, m_out,learning_rate):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        """
        
        self.input = None
        self.output = None
        self.Z = None
        self.delta = None
        
        self.W = None
        self.b = None
        self.p_learning_rate = learning_rate
        
         
        if self.W is None:
            self.W = np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + m_out)),
                    high=np.sqrt(6. / (n_in + m_out)),
                    size=(m_out,n_in)
                ),
                dtype=np.float32
            )

        if self.b is None:
            self.b = np.zeros((m_out,1), dtype=np.float32)
     
   
        
    def activation(self, W_x_b):
        dist = expit(W_x_b)
        #dist = 1./(1+np.exp(-W_x_b)); 
        return dist
    
    
    def derivation(self,X):
        derivative = X * (1 - X)
        return derivative
        
    def forwardPass(self,data):
        m = data.shape[1]
        b_expand = np.tile(self.b,(1,m))
        Z = np.dot(self.W,data) + b_expand
        A = self.activation(Z)
        self.input = data
        self.output = A
        self.Z = Z 
        return self.output
        
    def backwardPass(self,weighted_deltas):
        self.delta = weighted_deltas * self.derivation(self.output)
        return np.dot(self.W.T,self.delta)
        
    def updateGradient(self):
        for i in range(self.input.shape[1]):
            self.W = self.W - self.p_learning_rate * np.outer(self.delta[:,i],self.input[:,i])
            self.b = self.b - self.p_learning_rate * np.transpose(np.array([self.delta[:,i]]))
        