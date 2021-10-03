import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""
def loadpickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file,encoding='bytes')
    return data

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    meta_data_dict = loadpickle(data_dir+'/batches.meta')
    cifar_train_data = np.empty((0,3072))
    cifar_train_labels = []
    for i in range(1, 6):
        cifar_train_data_dict = loadpickle(data_dir + "/data_batch_{}".format(i))
        cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_labels += cifar_train_data_dict[b'labels']
    print(meta_data_dict)
    cifar_test_data_dict = loadpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_labels = cifar_test_data_dict[b'labels']
    x_train = cifar_train_data
    y_train = np.array(cifar_train_labels)
    x_test = cifar_test_data
    y_test = np.array(cifar_test_labels)
    ### YOUR CODE HERE
    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid

# if __name__ == '__main__':
#     xtr,ytr,xte,yte = load_data(data_dir='ResNet\cifar-10-python\cifar-10-batches-py')
#     print(xtr.shape,ytr.shape,xte.shape,yte.shape)