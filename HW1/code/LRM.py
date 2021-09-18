#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k
        
    def fit_BGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        print(labels)
        onehotlabels = np.eye(np.max(labels.astype(int))+1)[labels.astype(int)]
        np.random.seed(42)
        # self.W = 0.001*np.random.randn(X.shape[1],self.k)
        self.W = np.zeros((X.shape[1],self.k))
        n_samples, n_features = X.shape
        for i in range(self.max_iter):
            for j in range(0,n_samples,batch_size):
                if i ==1 and j == batch_size:
                    print('Weights after first epoch', self.W)
                # print(self.W)
                _g_total = 0
                if n_samples%batch_size != 0 and j/batch_size == int(n_samples/batch_size):
                    for k in range(batch_size*j,n_samples):
                        _g = self._gradient(X[k],onehotlabels[k])
                        _g_total = _g_total+_g
                    self.W = self.W + self.learning_rate*(-1*_g_total/(n_samples-batch_size*j))
                else:
                    for k in range(batch_size): #compute error over all batch
                        _g = self._gradient(X[j+k],onehotlabels[j+k])
                        _g_total = _g_total+_g
                    self.W = self.W + self.learning_rate*(-1*_g_total/batch_size)
                    # print(f'''iteartion {i} { np.linalg.norm(_g_total*1./batch_size)}''')
                if np.linalg.norm(_g_total*1./batch_size) < 0.0005:
                    print('breaking from loop convergence condition reached', i, ' value :',np.linalg.norm(_g_total*1./batch_size))
                    break
            else:
                continue
            break
        # print('onehotlabels',onehotlabels)
		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        x_new = _x[np.newaxis,:]
        y_new = _y - self.softmax(np.dot(self.W.T,_x))
        #y_new = _y - self.softmax(self.W.T@_x)
        y_new = y_new[np.newaxis,:]
        # print(x_new.shape,y_new.shape)
        _g =  -1*x_new.T @ y_new
        return _g
		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.
        return np.exp(x) / np.sum(np.exp(x), axis=0)
		### YOUR CODE HERE

		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        predict = self.softmax((X@self.W).T)
        preds = np.argmax(predict,axis=0)
        # print (predict,predict.shape)
        return preds
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        preds = self.predict(X)
        correct_classified = 0
        for i in range(X.shape[0]):
            if preds[i] == labels [i]:
                correct_classified = correct_classified + 1
        return (correct_classified/X.shape[0])*100
		### END YOUR CODE

