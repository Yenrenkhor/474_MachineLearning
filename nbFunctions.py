from sklearn.base import BaseEstimator
import numpy as np
import scipy.stats as stats

# For this assignment we will implement the Naive Bayes classifier as a
# a class, sklearn style. You only need to modify the fit and predict functions.
# Additionally, implement the Disparate Impact measure as the evaluateBias function.
class NBC(BaseEstimator):
    '''
    (a,b) - Beta prior parameters for the class random variable
    alpha - Symmetric Dirichlet parameter for the features
    '''

    def __init__(self, a=1, b=1, alpha=1):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.params = None
        
    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_alpha(self):
        return self.alpha

    # you need to implement this function

    def fit(self,X,y):
        '''
        This function does not return anything
        
        Inputs:
        X: Training data set (N x d numpy array)
        y: Labels (N length numpy array)
        '''
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        self.__classes = np.unique(y)

        # remove next line and implement from here
        # you are free to use any data structure for paramse
        params = None

        N = X.shape[0]
        
        d = X.shape[1]
        
        N1 = np.bincount(y)[1]
        
        theta = (N1 + a)/(N + a + b)
        
        #split up X1 and X2
        X1 = np.zeros((N1, d), dtype = int)
        X2 = np.zeros((N-N1, d), dtype = int)
        
        #iterate through X1 and X2 respectively
        k = 0
        l = 0
        for i in range(N):
            if y[i] == 1:
                X1[k] = X[i,:]
                k = k + 1
            else:
                X2[l] = X[i,:]
                l = l + 1
        
        print("done splitting X")
        
        #init counts of each feature
        am = np.unique(X).shape[0]
        N1j = np.zeros((am, d))
        N2j = np.zeros((am, d))
        K = np.zeros(d)
        theta1 = np.zeros((am, d))
        theta2 = np.zeros((am, d))
        
        for i in range(d):
            N1j[:,i] = np.bincount(X1[:,i], minlength = am)
            N2j[:,i] = np.bincount(X2[:,i], minlength = am)
            K[i] = np.unique(X[:, i]).shape[0]
            
            for j in range(am):
                if(N1j[j,i] != 0):
                    theta1[j,i] = (N1j[j,i] + alpha)/(N1 + K[i]*alpha)
                else:
                    theta1[j,i] = 0
                
                if(N2j[j,i] != 0):
                    theta2[j,i] = (N2j[j,i] + alpha)/(N-N1 + K[i]*alpha)
                else:
                    theta2[j,i] = 0
                
        print("done with counts")
        params = theta, theta1, theta2

        self.__params = params
    
    # you need to implement this function
    def predict(self,Xtest):
        '''
        This function returns the predicted class for a given data set
        
        Inputs:
        Xtest: Testing data set (N x d numpy array)
        
        Output:
        predictions: N length numpy array containing the predictions
        '''
        params = self.__params
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        #remove next line and implement from here
        #print(params)
        
        theta = params[0]
        theta1 = params[1]
        theta2 = params[2]
        
        predictions = np.random.choice(self.__classes,np.unique(Xtest.shape[0]))


        #do not change the line below
        return predictions
        
def evaluateBias(y_pred,y_sensitive):
    '''
    This function computes the Disparate Impact in the classification predictions (y_pred),
    with respect to a sensitive feature (y_sensitive).
    
    Inputs:
    y_pred: N length numpy array
    y_sensitive: N length numpy array
    
    Output:
    di (disparateimpact): scalar value
    '''
    #remove next line and implement from here
    di = 0
    
    #do not change the line below
    return di

def genBiasedSample(X,y,s,p,nsamples=1000):
    '''
    Oversamples instances belonging to the sensitive feature value (s != 1)
    
    Inputs:
    X - Data
    y - labels
    s - sensitive attribute
    p - probability of sampling unprivileged customer
    nsamples - size of the resulting data set (2*nsamples)
    
    Output:
    X_sample,y_sample,s_sample
    '''
    i1 = y == 1 # good
    i1 = i1[:,np.newaxis]
    i2 = y == 2 # bad
    i2 = i2[:,np.newaxis]
    
    sp = s == 1 #privileged
    sp = sp[:,np.newaxis]
    su = s != 1 #unprivileged
    su = su[:,np.newaxis]

    su1 = np.where(np.all(np.hstack([su,i1]),axis=1))[0]
    su2 = np.where(np.all(np.hstack([su,i2]),axis=1))[0]
    sp1 = np.where(np.all(np.hstack([sp,i1]),axis=1))[0]
    sp2 = np.where(np.all(np.hstack([sp,i2]),axis=1))[0]
    inds = []
    for i in range(nsamples):
        u = stats.bernoulli(p).rvs(1)
        if u == 1:
            #sample one bad instance with s != 1
            inds.append(np.random.choice(su2,1)[0])
            #sample one good instance with s == 1
            inds.append(np.random.choice(sp1,1)[0])
        else:
            #sample one good instance with s != 1
            inds.append(np.random.choice(su1,1)[0])
            #sample one bad instance with s == 1
            inds.append(np.random.choice(sp2,1)[0])
    X_sample = X[inds,:]
    y_sample = y[inds]
    s_sample = s[inds]
    
    return X_sample,y_sample,s_sample,inds
