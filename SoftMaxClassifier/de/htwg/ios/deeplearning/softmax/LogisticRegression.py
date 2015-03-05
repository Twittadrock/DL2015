'''
Created on Mar 4, 2015

@author: Matthias
'''

import numpy as np

class LogisticRegression(object):
    '''
    classdocs
    '''


    def __init__(self, n_in, m_out,learning_rate):
        '''
        Constructor
        '''
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
        w = np.array(W_x_b)
        maxes = np.amax(w, axis=0)
        #maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(w - maxes)
        sumi = np.sum(e, axis=0)
        disti = e / sumi
        return disti
    
    
    def derivation(self,X):
        return X * (1 - X)
    
    def forwardPass(self,data):
        m = data.shape[1]
        b_expand = np.tile(self.b,(1,m))
        Z = np.dot(self.W,data) + b_expand
        A = self.activation(Z)
        self.input = data
        self.output = A
        self.Z = Z 
        return self.output
        

       
    def predict(self):
        maxi = np.argmax(self.output, axis=0)
        return maxi#  
    
        
    def backwardPass(self,weighted_deltas):
        self.delta = weighted_deltas * self.derivation(self.output)
        return np.dot(self.W.T,self.delta)
        
    def updateGradient(self):
        for i in range(self.input.shape[1]):
            self.W = self.W - self.p_learning_rate * np.outer(self.delta[:,i],self.input[:,i])
            self.b = self.b - self.p_learning_rate * np.transpose(np.array([self.delta[:,i]]))
        
        
        