'''
Created on Mar 5, 2015

@author: Matthias
'''

import numpy as np

class ErrorLayer(object):
    '''
    classdocs
    '''


    def __init__(self,c_regularizer):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        """
        self.p_c_regularizer = c_regularizer
    
    def regularizer(self,layers):
        regularier = 0.
        for layer in layers:
            W = np.sum(np.absolute(layer.W))
            b = np.sum(np.absolute(layer.b))
            regularier += W + b
        return regularier
          
    def m_negative_log(self,y_labels,data):
        #y_bool = (y_labels == 1.)
        results = data.T[np.arange(len(data.T)), y_labels]
        return -np.mean(np.log(results))
   
    def errors(self,y_labels,data,layers=None):
        #results = self.output.T[np.arange(len(self.output.T)), y_labels]
        labels_encoded = np.zeros(data.shape)
        labels_encoded.T[np.arange(len(labels_encoded.T)), y_labels] = 1
        error = data - labels_encoded
        if layers is not None:
            error + self.p_c_regularizer * self.regularizer(layers)
        return error