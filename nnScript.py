'''
Neural Network Script Starts here
'''
from nnFunctions import *
import time
import matplotlib.pyplot as plt

# you may experiment with a small data set (mnist_sample.pickle) first
filename = 'mnist_all.pickle'
# filename = 'AI_quick_draw.pickle'
train_data, train_label, test_data, test_label = preprocess(filename)

#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
# n_hidden = 50
x = []
train = []
test = []
times = []
nlsp = np.linspace(5, 50, 10).astype(int)
for n_hidden in nlsp:
# set the number of nodes in output unit
    n_class = 10

# initialize the weights into some random matrices
    initial_W1 = initializeWeights(n_input, n_hidden)
    initial_W2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_W1.flatten(), initial_W2.flatten()), 0)

# set the regularization hyper-parameter

    lambdaval = 0

# args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
#
# # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
# opts = {'maxiter': 50}  # Preferred value.
#
# nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#
# # Reshape nnParams from 1D vector into W1 and W2 matrices
# W1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
# W2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
# print("training done!")

    st = time.time()
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    end = time.time()
    times.append(end - st)
    x.append(n_hidden)

    W1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    print("chicken done!")
    predicted_label = nnPredict(W1, W2, train_data)
    train.append(100 * np.mean((predicted_label == train_label).astype(float)))
    predicted_label = nnPredict(W1, W2, test_data)
    test.append(100 * np.mean((predicted_label == test_label).astype(float)))

fig = plt.figure(figsize=(20, 6))
ax = plt.subplot(2, 1, 1)
plt.plot(x, times)
ax.set_title('time')
ax = plt.subplot(2, 1, 2)
plt.plot(x, train, 'g', x, test, 'r')
ax.set_title('Accuracy')
plt.show()
# Test the computed parameters

# find the accuracy on Training Dataset
predicted_label = nnPredict(W1, W2, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# find the accuracy on Testing Dataset
predicted_label = nnPredict(W1, W2, test_data)
print('\n Test set Accuracy:    ' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

# obj = {'W1': W1, 'W2': W2, 'lambdaval':lambdaval,'n_hidden':n_hidden}
# pickle.dump(obj,open('params.pickle','wb'))