'''
Created on Mar 3, 2015

@author: Matthias
'''
import sys;
import theano
import numpy as np
import time
from theano import config
import sklearn.datasets as ds
from sklearn.decomposition import RandomizedPCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from LogisticRegression import LogisticRegression
from Hiddenlayer import Hiddenlayer
from ErrorLayer import ErrorLayer
from wx import PreDatePickerCtrl


class MLP(object):
    
    def __init__(self, learning_rate=0.1, learning_alpha = 0.99, epochs = 100, minibatch_size=50, hiddenlayers = [30 , 5], c_regularizer = 1):
        self.p_learning_rate = learning_rate
        self.p_learning_alpha = learning_alpha
        self.p_epochs = epochs
        self.p_minibatch_size = minibatch_size
        self.p_hiddenlayers = hiddenlayers
        self.p_c_regularizer = c_regularizer
        
        '''Network'''
        self.m_layers = []
        self.m_error_layer = None
        
        self.m_training_accuracy = []
        self.m_test_accuracy = []
        self.m_negative_log = []
        self.m_epochs = []
        
    def _forwardPass(self,X):
        message = X
        for layer in self.m_layers:
            message = layer.forwardPass(message)
        return message
    
    def _backwardPass(self,labels,O):
        delta = self.m_error_layer.errors(labels, O,self.m_layers)
        for layer in reversed(self.m_layers):
            delta = layer.backwardPass(delta)
    
    def _updateGradient(self):
        for layer in self.m_layers:
            layer.updateGradient()
    
    def _initLayers(self,in_n,out_m):
        '''
        Create m_layers
        '''
        for i in range(len(self.p_hiddenlayers)):
            if i==0:
                layer = Hiddenlayer(n_in=in_n,m_out=self.p_hiddenlayers[i],learning_rate=self.p_learning_rate)
            else:
                layer =  Hiddenlayer(self.p_hiddenlayers[i-1],m_out=self.p_hiddenlayers[i],learning_rate=self.p_learning_rate)     
            self.m_layers.append(layer)   
        output_layer = LogisticRegression(n_in=self.p_hiddenlayers[-1],m_out=out_m,learning_rate=self.p_learning_rate)
        self.m_layers.append(output_layer)
        self.m_error_layer = ErrorLayer(self.p_c_regularizer)
    
    def fit(self,X,y,X_test,y_test):
        '''
        Transpose because the layout of the net is horizontally
        '''
        X = X.T
        X_test = X_test.T
        classes = len(np.unique(y))
        self._initLayers(X.shape[0], classes)
        'Loop through m_epochs'
        self.m_epochs = range(self.p_epochs)
        for i in self.m_epochs:   
            batch_size = self.p_minibatch_size
            'Adjust learning rate'
            for layer in self.m_layers:
                layer.p_learning_rate = layer.p_learning_rate * self.p_learning_alpha
            for batch_n in xrange(0,X.shape[1],batch_size):
                image_batch = X[:,batch_n:batch_n+batch_size]
                label_batch = y[batch_n:batch_n+batch_size]
                'Forward Pass'
                O = self._forwardPass(image_batch)
                'Backward Pass'
                self._backwardPass(label_batch, O)
                'Update Gradient'
                self._updateGradient()
            print "Round #" + str(i)
            self._logEpoch(X,y,X_test,y_test)
    
    def plotResult(self):
        testplot, = plt.plot(self.m_epochs,self.m_test_accuracy, label='test')
        trainingplot,= plt.plot(self.m_epochs,self.m_training_accuracy, label='training')
        logplot, = plt.plot(self.m_epochs,self.m_negative_log, label='negative log x100')
        plt.legend([testplot, trainingplot, logplot], ['Test Accuracy', 'Training Accuracy', 'Negative Log x100'])
        plt.show()
    
    def _logEpoch(self, X,y,X_test,y_test):
        'Training'
        hits = 0
        misses = 0
        'Forward Pass'
        O = self._forwardPass(X)
        'Error'
        E = self.m_error_layer.m_negative_log(y, O)
        self.m_negative_log.append(E * 100)
        print "........Negative Log: " + str(E)
        pred =  self.m_layers[-1].predict()
        result = y - pred
        hits = hits + len(result[result == 0])
        misses = misses + len(result[result !=0] ) 
        hit_ratio = 100. * round(hits/float(len(result)),3)
        self.m_training_accuracy.append(hit_ratio)
        print "--------Training Hits " + str(hits) + " " + str(hit_ratio) + "%"
        'Test'
        hits = 0
        misses = 0
        'Forward Pass'
        self._forwardPass(X_test)
        pred =  self.m_layers[-1].predict()
        result = y_test - pred
        hits = hits + len(result[result == 0])
        misses = misses + len(result[result !=0] ) 
        hit_ratio = 100. * round(hits/float(len(result)),3)
        self.m_test_accuracy.append(hit_ratio)
        print "--------Testing  Hits " + str(hits)+ " " + str(hit_ratio) + "%"
    
    def predict(self,X):
        X = X.T
        self._forwardPass(X)
        pred =  self.m_layers[-1].predict()
        return pred
    
    
    def loadSampleDate(self):
        digits = ds.load_digits()
        #print(digits.DESCR)
        iris = ds.load_iris()
        X, y = digits.data, digits.target
        print "Images Shape: " + str(X.shape)
        
        classes_m = len(np.unique(y))
        print "Classes: " + str(classes_m)
            
        training_count = int(X.shape[0] * 0.8)
        test_count = int(X.shape[0] * 0.2)
        
        subsample = np.random.permutation(X.shape[0])  
        training = X[subsample[:training_count]]
        training_labels = y[subsample[:training_count]]
        test = X[subsample[training_count:]]
        test_labels = y[subsample[training_count:]]
        print "Training Images: " + str(training.shape)
        print "Test Images: " + str(test.shape)
        return training, training_labels, test, test_labels

def plot_gallery(data, labels, shape, interpolation='nearest'):
    for i in range(data.shape[0]):
        plt.subplot(1, data.shape[0], (i + 1))
        plt.imshow(data[i].reshape(shape), interpolation=interpolation)
        plt.title(labels[i])
        plt.xticks(()), plt.yticks(())
    plt.show()
    return plt
        
if __name__ == '__main__':
    
    mlp = MLP(learning_rate=0.1, learning_alpha=0.99, epochs=100, minibatch_size=50, hiddenlayers = [30 , 5],c_regularizer = 1)
    training, training_labels, test, test_labels = mlp.loadSampleDate()
    mlp.fit(training, training_labels, test, test_labels)
    mlp.plotResult()
    print "Predicted:  " + str(mlp.predict(test[:10]))
    print "True class: " + str(test_labels[:10])
    plot_gallery(test[:10], test_labels[:10], shape=(8, 8))
    pass