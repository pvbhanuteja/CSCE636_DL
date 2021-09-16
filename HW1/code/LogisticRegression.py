import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_GD(self, X, y):
        """Train perceptron model on data (X,y) with GD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        weights = np.random.normal(0, 0.1, n_features) # 0 mean and 0.1 sigma
        self.assign_weights(weights)
        for i in range(self.max_iter):
            w_error = 0
            for j in range(n_samples): #compute error over all samples
                _g = self._gradient(X[j],y[j])
                w_error = w_error+_g
            self.W = self.W + self.learning_rate*(-1*w_error/n_samples)
		### END YOUR CODE
        return self

    def fit_BGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
		### YOUR CODE HERE
        weights = np.random.normal(0, 0.1, n_features) # 0 mean and 0.1 sigma
        self.assign_weights(weights)
        for i in range(self.max_iter):
            # w_error = 0
            ## CODE for randoly taking batch size samples and training
            # batch_indexes = np.random.choice(range(n_samples), batch_size, replace=False)
            # for j in batch_indexes: #compute error over all samples
            #     _g = self._gradient(X[j],y[j])
            #     w_error = w_error+_g
            # self.W = self.W + self.learning_rate*(-1*w_error/batch_size)
            for j in range(0,n_samples,batch_size):
                w_error = 0
                for k in range(batch_size): #compute error over all batch
                    _g = self._gradient(X[j+k],y[j+k])
                    w_error = w_error+_g
                self.W = self.W + self.learning_rate*(-1*w_error/batch_size)
		### END YOUR CODE
            
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with SGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        
        weights = np.random.normal(0, 0.1, n_features) # 0 mean and 0.1 sigma
        self.assign_weights(weights)
        for i in range(self.max_iter):
            random_sample_pos = np.random.randint(low=0, high=n_samples-1)
            _g = self._gradient(X[random_sample_pos],y[random_sample_pos])
            self.W = self.W + self.learning_rate*(-1*_g)
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        _g = -1*_y*_x*(1/(1+(np.exp(_y*np.dot(self.W.T,_x)))))
        return _g
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

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		## YOUR CODE HERE
        # verti_cloned_w = np.tile(self.W,(X.shape[0],1))
        # print(self.W)
        # print("verti_cloned_w",verti_cloned_w.shape)
        XmulW = np.dot(X,self.W)
        yeq1pred = 1/(1+np.exp(-1*XmulW))
        preds_proba = np.concatenate(([yeq1pred],[1-yeq1pred]),axis=0).T
        return preds_proba
		## END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        predictions = np.ones(X.shape[0])
        pred_probs = self.predict_proba(X)
        for i in range(X.shape[0]):
            if pred_probs[i][0]>pred_probs[i][1]:
                predictions[i] = 1
            else:
                predictions[i] = -1
        print("Uniques->",np.unique(predictions, return_counts=True))
        return predictions
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        predictions = self.predict(X)
        correct_classified = 0
        for i in range(X.shape[0]):
            if predictions[i] == y [i]:
                correct_classified = correct_classified + 1

        return (correct_classified/X.shape[0])*100

		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

