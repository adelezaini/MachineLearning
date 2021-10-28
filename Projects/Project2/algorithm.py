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
    
    
def learning_schedule(t, t0=5, t1=50):
    return t0/(t+t1)
    
def SGD(X, y, lmd, gradient, n_epochs, m, t0=5, t1=50):
    """Stochastic Gradient Descent Algorithm
    
        Args:
        - X (array): design matrix (training data)
        - y (array): output dataset (training data)
        - gradient (function): function to compute the gradient
        - n_epochs (int): number of epochs
        - m (int): number of minibatches
        - t0 (float): initial paramenter to compute the learning rate
        - t1 (float): sequential paramenter to compute the learning rate
        
        Returns:
        beta/theta-values"""
        
    theta = np.random.randn(X.shape[1])
    
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            Xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = gradient(Xi, yi, theta, lmd) #* X.shape[0] #2.0 * Xi.T @ ((Xi @ theta)-yi)
            eta = learning_schedule(epoch*m+i, t0=t0, t1=t1)
            theta = theta - eta*gradients
            
    return theta
    


