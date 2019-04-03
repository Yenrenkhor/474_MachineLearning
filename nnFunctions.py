import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() and preprocess()
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer
    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer
    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    '''
     Input:
     filename: pickle file containing the data_size
     scale: scale data to [0,1] (default = True)
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    '''
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label

def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # your code here - remove the next four lines

    s = 1/(1 + np.exp(-1 * z))
    return s

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of nodes in input layer (not including the bias node)
    % n_hidden: number of nodes in hidden layer (not including the bias node)
    % n_class: number of nodes in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2
    n_data = train_data.shape[0]

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    #1 of k
    label_mat = np.zeros((n_data, n_class))
    label_mat[range(n_data), train_label] = 1

    bias = np.ones((train_data.shape[0], 1), dtype=int)
    train_data = np.hstack((train_data, bias))


    a = np.dot(W1, train_data.T)
    hid_1 = sigmoid(a)
    bias_2 = np.ones((1, train_data.shape[0]), dtype=int)
    input_2 = np.vstack((hid_1, bias_2))
    net = np.dot(W2, input_2)
    o = sigmoid(net).T
    error = -np.sum(np.multiply(label_mat, np.log(o))+np.multiply((1-label_mat), np.log(1-o)), axis=1)
    obj_val = np.sum(error)/train_data.shape[0]

    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0)

    delta = o - label_mat
    djw2 = np.matmul(delta.T, input_2.T) / n_data

    W2noBias = W2[:, 0:-1]
    z = np.multiply((1 - hid_1), hid_1)
    err = np.multiply(np.dot(delta, W2noBias), z.T)
    djw1 = np.matmul(np.multiply(z, err.T), train_data) / n_data  # this value is wrong
    obj_grad = np.hstack((djw1.flatten(), djw2.flatten()))
    return (obj_val, obj_grad)


def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.
    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image
    % Output:
    % label: a column vector of predicted labels
    '''

    bias = np.ones((data.shape[0], 1), dtype=int)
    data = np.hstack((data, bias))

    a = np.dot(W1, data.T)
    hid_1 = sigmoid(a)
    bias_2 = np.ones((1, data.shape[0]), dtype=int)
    input_2 = np.vstack((hid_1, bias_2))
    
    net = np.dot(W2, input_2)
    o = sigmoid(net).T
    labels=np.argmax(o,axis=1)
    return labels