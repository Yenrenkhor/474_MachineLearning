from nnFunctions import *


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

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    label_mat = np.zeros((train_label.shape[0], n_class))
    label_mat[range(train_label.shape[0]), train_label] = 1

    bias = np.ones((train_data.shape[0], 1), dtype=int)
    train_data = np.hstack((train_data, bias))


    a = np.dot(W1, train_data.T)
    hid_1 = sigmoid(a)
    bias_2 = np.ones((1, train_data.shape[0]), dtype=int)
    input_2 = np.vstack((hid_1, bias_2))
    # print(input_2.shape)
    # print(W2.shape)
    net = np.dot(W2, input_2)
    o = sigmoid(net).T
    ologjohnson=np.log(o)
    j1=np.multiply(label_mat,ologjohnson)
    j2=np.multiply((1-label_mat),np.log(1-o))
    error=-np.sum(j1+j2,axis=1)
    obj_val=np.sum(error)/train_data.shape[0]

    delta = o - label_mat
    djw2 = np.matmul(delta.T, input_2.T)
    print(djw2)

    W2noBias = W2[:, 0:50]
    coeff = np.multiply((1 - hid_1), hid_1)
    djw1 = np.dot(delta, W2)
    #obj_grad = np.hstack()

    output = np.array(np.zeros(n_class))



    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0)
    obj_grad = np.zeros(params.shape)

    return (obj_val, obj_grad)



n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 0
params = np.linspace(-5,5, num=26)
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
objval,objgrad = nnObjFunction(params, *args)
print("Objective value:")
print(objval)
print("Gradient values: ")
print(objgrad)