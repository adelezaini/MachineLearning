# The MIT License (MIT)
#
# Copyright © 2021 Adele Zaini
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
# SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import numpy as np
from random import random, seed
from autograd import grad

def SVD(A):
    """ Application of SVD theorem.
    Useful for debugging. """
    U, S, VT = np.linalg.svd(A,full_matrices=True)
    D = np.zeros((len(U),len(VT)))
    print("shape D= ", np.shape(D))
    print("Shape S= ",np.shape(S))
    print("lenVT =",len(VT))
    print("lenU =",len(U))
    D = np.eye(len(U),len(VT))*S
    """
    for i in range(0,VT.shape[0]): #was len(VT)
        D[i,i]=S[i]
        print("i=",i)"""
    return U @ D @ VT
    
def SVDinv(A):
    """Evaluate the inverse of a matrix using the SVD theorem"""
    U, s, VT = np.linalg.svd(A)
    # reciprocals of singular values of s
    d = 1.0 / s
    # create m x n D matrix
    D = np.zeros(A.shape)
    # populate D with n x n diagonal matrix
    D[:A.shape[1], :A.shape[1]] = np.diag(d)
    UT = np.transpose(U)
    V = np.transpose(VT)
    return np.matmul(V,np.matmul(D.T,UT))
    

optimizers = ["SGD", "ADAGRAD", "RMS", "ADAM"]
eta_types = ['static', 'schedule', 'invscaling', 'hessian']

def GD(X, y, lmd, gradient, eta = 0.1, Niterations = 1000):
    """Gradient Descent Algorithm
    
        Args:
        - X (array): design matrix (training data)
        - y (array): output dataset (training data)
        - gradient (function): function to compute the gradient
        - eta (float): learning rate
        - Niterations (int): number of iteration
        
        Returns:
        beta/theta-values"""
    theta = np.random.randn(X.shape[1])

    for iter in range(Niterations):
        gradients = gradient(X, y, theta, lmd) #2.0/X.shape[0] * X.T @ ((X @ theta) - y) #
        theta -= eta*gradients
        
    return theta
    
def gradient_of(function, *args, index = 0): # not tested
    """ Evaluate the gradient of the 'function' with the given 'arg*', using autograd.
        The 'index' stands for the variable to derivate on (first one = 0,second one = 1, ...).
        NB: the size of '*args' depends on the function itself. """
    gradient = grad(function, index)
    return gradient(args)
    
    
def learning_schedule(t, t0=5, t1=50):
    return t0/(t+t1)
    
def SGD(X, y, lmd, gradient, n_epochs, M, opt = "SGD", eta0 = 0.1, eta_type = 'schedule', t0=5, t1=50, momentum = 0., rho = 0.9, b1 = 0.9, b2 = 0.999):
    """Stochastic Gradient Descent Algorithm
    
        Args:
        - X (array): design matrix (training data)
        - y (array): output dataset (training data)
        - gradient (function): function to compute the gradient
        - n_epochs (int): number of epochs
        - M (int): size of minibatches
        - opt (string): "SGD", "ADAGRAD", "RMS", "ADAM" - different optimizers
        - eta0 (float): learning rate if 'static' or 'invscaling'
        - eta_type = 'static', 'schedule', 'invscaling', 'hessian' - different methods for evaluating the learning rate
        - t0 (float): initial paramenter to compute the learning rate in 'schedule'
        - t1 (float): sequential paramenter to compute the learning rate in 'schedule'
        - momentum, rho, b1, b2 (float): parameters for different optimizers
        
        Returns:
        beta/theta-values"""
        
    if opt not in optimizers:
        raise ValueError("Optimizer must be defined in "+str(optimizers))
        
    if eta_type not in eta_types:
        raise ValueError("Learning rate type must be defined within "+str(eta_types))
        
    theta = np.random.randn(X.shape[1])
    m = int(X.shape[0]/M)
    v = np.zeros(X.shape[1]) # parameter for velocity (momentum), squared-gradient (adagrad, RMS),
    ma = np.zeros(X.shape[1]) # parameter for adam
    delta = 1e-1
              
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X[random_index:random_index + M]
            yi = y[random_index:random_index + M]
            gradients = gradient(Xi, yi, theta, lmd) #* X.shape[0] #2.0 * Xi.T @ ((Xi @ theta)-yi)
            
            # Evaluate the hessian metrix to test eta < max H's eigvalue
            H = (2.0/X.shape[0])* (X.T @ X)
            eigvalues, eigvects = np.linalg.eig(H)
            eta_opt = 1.0/np.max(eigvalues)
            eta = eta_opt
            
            if eta_type == 'static':
                eta = eta0
            elif eta_type == 'schedule':
                eta = learning_schedule(epoch*m+i, t0=t0, t1=t1)
            elif eta_type == 'invscaling':
                power_t = 0.25 # one can change it but I dont want to overcrowd the arguments
                eta = eta0 / pow(epoch*m+i, power_t)
            elif eta_type == 'hessian':
                pass
                
            assert eta > eta_opt, "Learning rate higher than the inverse of the max eigenvalue of the Hessian matrix: SGD will not converge to the minimum. Need to set another learning rate or its paramentes."
            
            if opt == "SDG":
                v = momentum * v - eta * gradients
                theta = theta + v
            elif opt == "ADAGRAD":
                v = v + np.multiply(gradients, gradients)
                theta = theta - np.multiply(eta / np.sqrt(v+delta), gradients)
            elif opt == "RMS":
                v = rho * v + (1. - rho) * np.multiply(gradients, gradients)
                theta = theta - np.multiply(eta / np.sqrt(v+delta), gradients)
            elif opt == "ADAM":
                ma = b1 * ma + (1. - b1) * gradients
                v = b2 * v + (1. - b2) * np.multiply(gradients, gradients)
                ma = ma / (1. - b1)
                v = v / (1. - b2)
                theta = theta - np.multiply(eta / np.sqrt(v+delta), ma)
                
    return theta
    


