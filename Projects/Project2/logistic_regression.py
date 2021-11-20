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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as sk #import SGDRegressor, Lasso, LinearRegression, Ridge
from dataset import train_test_rescale
from misc import MSE, R2
from algorithm import SVDinv, SGD, GD
from sklearn.utils import resample
from activation import *
from regression import *
    

class LogisticRegression:
    """This class implements the Logistic Regression, as classification problem.
        The Standard Gradient Descent has been implemented."""
        
    def __init__(self, X, y):
        """Initialize the class.
        Args:
          - X : n_datapoints with features to train, shape (n_feautures, n_datapoints)
          - y : n_labels of all datapoints, shape (n_labels,)"""
        self.X = X
        self.y = y
    
    def split(self,test_size=0.2):
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size)
        return self
        
    def loss(self, y, y_hat):
        return -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    
    def gradients(self, X, y, y_hat):
        
        # X --> Input.
        # y --> true/target value.
        # y_hat --> hypothesis/predictions.
        # w --> weights (parameter).
        # b --> bias (parameter).
        
        # m-> number of training examples.
        m = X.shape[0]
        
        # Gradient of loss w.r.t weights.
        dw = (1/m)*np.dot(X.T, (y_hat - y))
        
        # Gradient of loss w.r.t bias.
        db = (1/m)*np.sum((y_hat - y))
        
        return dw, db
        
        
    def normalize(self, X):
    
        # X --> Input.
        
        # m-> number of training examples
        # n-> number of features
        m, n = X.shape
        
        # Normalizing all the n features of X.
        for i in range(n):
            X = (X - X.mean(axis=0))/X.std(axis=0)
            
        return X
        
    def train(self, X, y, bs, epochs, eta):
        
        # X --> Input.
        # y --> true/target value.
        # bs --> Batch Size.
        # epochs --> Number of iterations.
        # eta --> Learning rate.
            
        # m-> number of training examples
        # n-> number of features
        m, n = X.shape
        
        # Initializing weights and bias to zeros.
        w = np.zeros((n,1))
        b = 0
        
        # Reshaping y.
        y = y.reshape(m,1)
        
        # Normalizing the inputs.
        x = normalize(X)
        
        # Empty list to store losses.
        losses = []
        
        # Training loop.
        for epoch in range(epochs):
            for i in range((m-1)//bs + 1):
                
                # Defining batches. SGD.
                start_i = i*bs
                end_i = start_i + bs
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]
                
                # Calculating hypothesis/prediction.
                y_hat = sigmoid(np.dot(xb, w) + b)
                
                # Getting the gradients of loss w.r.t parameters.
                dw, db = gradients(xb, yb, y_hat)
                
                # Updating the parameters.
                w -= lr*dw
                b -= lr*db
            
            # Calculating loss and appending it in the list.
            l = loss(y, sigmoid(np.dot(X, w) + b))
            losses.append(l)
            
        # returning weights, bias and losses(List).
        return w, b, losses
        
        
    def predict(self, X):
        
        # X --> Input.
        
        # Normalizing the inputs.
        x = normalize(X)
        
        # Calculating presictions/y_hat.
        preds = sigmoid(np.dot(X, w) + b)
        
        # Empty List to store predictions.
        pred_class = []    # if y_hat >= 0.5 --> round up to 1
        # if y_hat < 0.5 --> round up to 1
        pred_class = [1 if i > 0.5 else 0 for i in preds]
        
        return np.array(pred_class)
        
    def accuracy(self, y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracyaccuracy(X, y_hat=predict(X))
