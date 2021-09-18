import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import ptp
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    one_labels = X[np.where(y==1)]
    minusone_labels = X[np.where(y==-1)]
    plt.scatter(one_labels[:,0], one_labels[:,1], c="g", alpha=1, marker="o",label="1 Lables")
    plt.scatter(minusone_labels[:,0], minusone_labels[:,1], c="r", alpha=1, marker="+",label="-1 Lables")
    plt.xlabel("Feature 1(Symmentry)")
    plt.ylabel("Feature 2(Avg. Intensity)")
    plt.legend(loc='lower right')
    plt.show()
    # print(X,X.shape)
    # print(y,y.shape)
    # print(one_labels[:,1],one_labels.shape)
    # print(minusone_labels,minusone_labels.shape)
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_features,].
	
	Returns:
		No return. Save the plot to 'train_result_sigmoid.*' and include it
		in submission.
	'''
	### YOUR CODE HERE
    # one_labels = X[np.where(y==1)]
    # minusone_labels = X[np.where(y==-1)]
    # plt.scatter(one_labels[:,0], one_labels[:,1], c="g", alpha=1, marker="o",label="1 Lables")
    # plt.scatter(minusone_labels[:,0], minusone_labels[:,1], c="r", alpha=1, marker="+",label="-1 Lables")
    # xlimit = np.array(plt.gca().get_xlim())
    # print('xlimit',xlimit)
    # m = -W[1]/W[2]
    # c = -W[0]/W[2]
    # print("m",m,"\n","c",c)
    # plt.gca().set_ylim((-1,0))
    # plt.plot(xlimit, m * xlimit + c )
    # plt.xlabel("Feature 1(Symmentry)")
    # plt.ylabel("Feature 2(Avg. Intensity)")
    # plt.legend(loc='lower right')
    # plt.show()
	# ### END YOUR CODE
    one_labels = X[np.where(y==1)]
    minusone_labels = X[np.where(y==-1)]
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .002  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = np.c_[xx.ravel(), yy.ravel()]
    X = np.concatenate((np.ones((XX.shape[0],1)),XX),axis=1)
    XmulW = np.dot(X,W)
    yeq1pred = 1/(1+np.exp(-1*XmulW))
    pred_probs = np.concatenate(([yeq1pred],[1-yeq1pred]),axis=0).T
    print(pred_probs)
    Z = np.ones(X.shape[0])
    for i in range(X.shape[0]):
        if pred_probs[i][0]>pred_probs[i][1]:
            Z[i] = 1
        else:
            Z[i] = -1
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    # plt.scatter(K[:, 0], K[:, 1], edgecolors='k', cmap=plt.cm.Paired)
    plt.scatter(one_labels[:,0], one_labels[:,1], c="g", alpha=1, marker="o",label="1 Lables")
    plt.scatter(minusone_labels[:,0], minusone_labels[:,1], c="r", alpha=1, marker="+",label="-1 Lables")
    plt.xlabel("Feature 1(Symmentry)")
    plt.ylabel("Feature 2(Avg. Intensity)")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())

    plt.show()

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 
	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 0,1,2.
		W: An array of shape [n_features, 3].
	
	Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
	'''
    ### YOUR CODE
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    one_labels = X[np.where(y==1)]
    two_labels = X[np.where(y==2)]
    zero_labels = X[np.where(y==0)]
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .002  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = np.c_[xx.ravel(), yy.ravel()]
    X = np.concatenate((np.ones((XX.shape[0],1)),XX),axis=1)
    predict = softmax((X@W).T)
    Z = np.argmax(predict,axis=0)
    Z = Z.reshape(xx.shape)
    # plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    # plt.scatter(K[:, 0], K[:, 1], edgecolors='k', cmap=plt.cm.Paired)
    plt.scatter(one_labels[:,0], one_labels[:,1], c="g", alpha=1, marker="o",label="1 Lables")
    plt.scatter(two_labels[:,0], two_labels[:,1], c="r", alpha=1, marker="+",label="2 Lables")
    plt.scatter(zero_labels[:,0], zero_labels[:,1], c="b", alpha=1, marker="+",label="0 Lables")
    plt.xlabel("Feature 1(Symmentry)")
    plt.ylabel("Feature 2(Avg. Intensity)")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())

    plt.show()
    # plt.xlabel("Feature 1(Symmentry)")
    # plt.ylabel("Feature 2(Avg. Intensity)")
    # plt.legend(loc='lower right')
    # plt.show()
	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

    ## Visualize training data.
    #uncomment
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check GD, SGD, BGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_GD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ## YOUR CODE HERE
    # Trying with different learning rates
    learning_rates = [0.1,0.5,1,0.01,0.001]
    score_learning_rate =[]
    for _lrnrate in learning_rates:
        logisticR_classifier = logistic_regression(learning_rate=_lrnrate, max_iter=100)
        logisticR_classifier.fit_BGD(train_X, train_y, 10)
        score_learning_rate.append(logisticR_classifier.score(valid_X, valid_y))
    plt.scatter(learning_rates,score_learning_rate)
    plt.xlabel("learning_rates")
    plt.ylabel("Validation scores")
    plt.show()
    score_batch = []
    exp_batches = [5,10,50,100,300]
    for _batchsize in exp_batches:
        logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)
        logisticR_classifier.fit_BGD(train_X, train_y, _batchsize)
        score_batch.append(logisticR_classifier.score(valid_X, valid_y))
    plt.scatter(exp_batches,score_batch)
    plt.xlabel("Batch size")
    plt.ylabel("Validation scores")
    plt.show()
    

    # Best model with 5 batch size and 0.5 LR
    best_logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)
    # To include train and valdiation data for final training
    best_logisticR_classifier.fit_BGD(np.concatenate((train_X,valid_X),axis=0), np.concatenate([train_y,valid_y]), 5)
    # best_logisticR_classifier.fit_BGD(train_X,train_y, 30)
    print("score====>",best_logisticR_classifier.score(valid_X, valid_y))
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    # visualize_result(train_X[:, 1:3], train_y, best_logisticR_classifier.get_params())

    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, best_logisticR_classifier.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    raw_data, labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(raw_data)
    test_y_all, test_idx = prepare_y(labels)
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y==2)] = -1
    print('Test score on 5 Batch size and 0.5LR sigmoid model \n',best_logisticR_classifier.score(test_X, test_y))
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  BGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.predict(train_X))
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    learning_rates = [0.1,0.5,1,0.01,0.001]
    score_learning_rate =[]
    for _lrnrate in learning_rates:
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=_lrnrate, max_iter=100,  k= 3)
        logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
        score_learning_rate.append(logisticR_classifier_multiclass.score(valid_X, valid_y))
    plt.scatter(learning_rates,score_learning_rate)
    plt.xlabel("learning_rates")
    plt.ylabel("Validation scores")
    plt.show()
    score_batch = []
    exp_batches = [5,10,50,100,300]
    for _batchsize in exp_batches:
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
        logisticR_classifier_multiclass.fit_BGD(train_X, train_y, _batchsize)
        score_batch.append(logisticR_classifier_multiclass.score(valid_X, valid_y))
    plt.scatter(exp_batches,score_batch)
    plt.xlabel("Batch size")
    plt.ylabel("Validation scores")
    plt.show()
    ### END YOUR CODE
    best_logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    best_logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
	# Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, best_logisticR_classifier_multiclass.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    # print(test_X_all.shape,test_y_all.shape)
    print("Test score on 10 Batch size and 0.5LR softmax model \n",best_logisticR_classifier_multiclass.score(test_X_all, test_y_all))
    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=1000,  k= 2)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(valid_X, valid_y))
    ### END YOUR CODE

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=1000)
    logisticR_classifier.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(valid_X, valid_y))
    ## END YOUR CODE


    ################Compare and report the observations/prediction accuracy


# '''
# Explore the training of these two classifiers and monitor the graidents/weights for each step. 
# Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
# Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
# '''
    ### YOUR CODE HERE
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1 

    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)
    logisticR_classifier.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(valid_X, valid_y))

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0   

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.25, max_iter=1000,  k= 2)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(valid_X, valid_y))
    ### END YOUR CODE

    # ------------End------------


if __name__ == '__main__':
    main()
    
    
