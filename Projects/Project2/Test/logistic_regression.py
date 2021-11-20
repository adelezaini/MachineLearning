

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
 
class LogisticRegression:
# source: https://github.com/michelucci/Logistic-Regression-Explained/blob/master/MNIST%20with%20Logistic%20Regression%20from%20scratch.ipynb
    def sigmoid(z):
        """
        Implement the sigmoid function

        Arguments:
        y -- a scalar (float)

        Return:
        s -- the sigmoid function evaluated on z (as in equation (1))
        """
        s = 1.0 / (1.0 + np.exp(-z))
        
        return s
        
    def initialize(dim):
        """
        Initialise the weights and the bias to tensors of dimensions (dim,1) for w and
        to 1 for b (a scalar)

        Arguments:
        dim -- a scalar (float)

        Return:
        w -- a matrix of dimensions (dim,1) containing all zero
        b -- a scalar = 0
        """
        w = np.zeros((dim,1))
        b = 0
        
        assert (w.shape == (dim,1))
        assert (isinstance(b, float) or isinstance(b,int))
        
        return w,b
        
    def propagate(w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px, 1) (our case 784,1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px, number of examples)
        Y -- true "label" vector (containing 0 if class 1, 1 if class 2) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """
        
        m = X.shape[1]
        
        z = np.dot(w.T,X)+b
        A = sigmoid(z)
        cost = -1.0/m*np.sum(Y*np.log(A)+(1.0-Y)*np.log(1.0-A))
        
        dw = 1.0/m*np.dot(X, (A-Y).T)
        db = 1.0/m*np.sum(A-Y)
        
        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        
        grads = {"dw": dw,
                 "db":db}
        
        return grads, cost
        
    def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (n_x, 1)
        b -- bias, a scalar
        X -- data of shape (n_x, m)
        Y -- true "label" vector (containing 0 if class 1, 1 if class 2), of shape (1, m)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        costs = []
        
        for i in range(num_iterations):
            
            grads, cost = propagate(w, b, X, Y)
            
            dw = grads["dw"]
            db = grads["db"]
            
            w = w - learning_rate*dw
            b = b - learning_rate*db
            
            if i % 100 == 0:
                costs.append(cost)
                
            if print_cost and i % 100 == 0:
                print ("Cost (iteration %i) = %f" %(i, cost))
                
        grads = {"dw": dw, "db": db}
        params = {"w": w, "b": b}
            
        return params, grads, costs
        
    def predict (w, b, X):
        '''Predict whether the label is 0 or 1
        
        Arguments:
        w -- weights, a numpy array of size (n_x, 1)
        b -- bias, a scalar
        X -- data of size (n_x, m)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1)
        '''
        
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0],1)
        
        A = sigmoid (np.dot(w.T, X)+b)
        
        for i in range(A.shape[1]):
            if (A[:,i] > 0.5):
                Y_prediction[:, i] = 1
            elif (A[:,i] <= 0.5):
                Y_prediction[:, i] = 0
                
        assert (Y_prediction.shape == (1,m))
        
        return Y_prediction
        
    def fit(self, X_train, Y_train, X_test, Y_test, num_iterations = 1000, learning_rate = 0.5, print_cost = False):
    
        w, b = initialize(X_train.shape[0])
        parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        
        w = parameters["w"]
        b = parameters["b"]
        
        Y_prediction_test = predict (w, b, X_test)
        Y_prediction_train = predict (w, b, X_train)
        
        train_accuracy = 100.0 - np.mean(np.abs(Y_prediction_train-Y_train)*100.0)
        test_accuracy = 100.0 - np.mean(np.abs(Y_prediction_test-Y_test)*100.0)
        
        self.d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}
        
        print ("Accuarcy Test: ",  test_accuracy)
        print ("Accuracy Train: ", train_accuracy)
        
        return self.d
        
    def plot_cost(self):
        plt.plot(d["costs"])
        plt.title("Training loss",fontsize = 15)
        plt.xlabel("Number of iterations (1e2)", fontsize = 14)
        plt.ylabel("$Cross-entropy$", fontsize = 17)
        plt.show()
    
class LogisticRegression3:
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
        
    def fitGD(self, eta, epochs):
    
        # Initialize the weights and bias i.e. 'm' and 'c'
        w = np.zeros_like(X[0]) # array with shape equal to no. of features
        b = 0
        m = X.shape[0]

        # Define sigmoid function
        def sigmoid(z):
         return 1./(1+np.exp(-z))
         
        # Performing Gradient Descent Optimization
        # for every epoch
        for epoch in range(1,epochs+1):
            # for every data point(X_train,y_train)
            for i in range(len(X)):
                #compute gradient for weights and biases
                dw = X[i] * (y[i] - self.sigmoid(np.dot(w.T, X[i]) + b))
                db = y[i] - self.sigmoid(np.dot(w.T, X[i]) + b)
                #update m, c
                dw = w - eta * dw
                db = b - eta * db
        # At the end of all epochs we will be having optimum values of 'm' and 'c'
        # So by using those optimum values of 'm' and 'c' we can perform predictions
        predictions = []
        for i in range(len(X)):
         z = np.dot(m, X[i]) + c
         y_pred = sigmoid(z)
         if y_pred>=0.5:
          predictions.append(1)
         else:
          predictions.append(0)
        # 'predictions' list will contain all the predicted class labels using optimum 'm' and 'c'


class LogisticRegression2:
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
        
    def loss(y, y_hat):
        return -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    
    def gradients(X, y, y_hat):
        
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
        
    def plot_decision_boundary(X, w, b):
    
        # X --> Inputs
        # w --> weights
        # b --> bias
        
        # The Line is y=mx+c
        # So, Equate mx+c = w.X + b
        # Solving we find m and c
        x1 = [min(X[:,0]), max(X[:,0])]
        m = -w[0]/w[1]
        c = -b/w[1]
        x2 = m*x1 + c
        
        # Plotting
        fig = plt.figure(figsize=(10,8))
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
        plt.xlim([-2, 2])
        plt.ylim([0, 2.2])
        plt.xlabel("feature 1")
        plt.ylabel("feature 2")
        plt.title('Decision Boundary')
        plt.plot(x1, x2, 'y-')
        
    def normalize(X):
    
        # X --> Input.
        
        # m-> number of training examples
        # n-> number of features
        m, n = X.shape
        
        # Normalizing all the n features of X.
        for i in range(n):
            X = (X - X.mean(axis=0))/X.std(axis=0)
            
        return X
        
    def train(X, y, bs, epochs, lr):
        
        # X --> Input.
        # y --> true/target value.
        # bs --> Batch Size.
        # epochs --> Number of iterations.
        # lr --> Learning rate.
            
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
        
        
    def predict(X):
        
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
        
    def accuracy(y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracyaccuracy(X, y_hat=predict(X))
"""
w, b, l = train(X, y, bs=100, epochs=1000, lr=0.01)# Plotting Decision Boundary
plot_decision_boundary(X, w, b)
accuracy(y, predict(X))"""
 
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse=False)

class MultiClassifier:

    def __init__(self, X, y):
        """Initialize the class.
        Args:
          - X : n_datapoints with features to train, shape (n_feautures, n_datapoints)
          - y : n_labels of all datapoints, shape (n_labels,)"""
        self.X = X
        self.y = y

    def loss(X, Y, W):
        """Y: onehot shape"""
        Z = - X @ W
        N = X.shape[0]
        loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def gradient(X, Y, W, mu):
        """Y: onehot shape"""
        Z = - X @ W
        P = softmax(Z, axis=1)
        N = X.shape[0]
        gd = 1/N * (X.T @ (Y - P)) #+ 2 * mu * W
        return gd

    def fit_GD(X, Y, max_iter=1000, eta=0.1, mu=0.01):
        """Very basic gradient descent algorithm with fixed eta and mu"""
        Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
        W = np.zeros((X.shape[1], Y_onehot.shape[1]))
        step = 0
        step_lst = []
        loss_lst = []
        W_lst = []
     
        while step < max_iter:
            step += 1
            W -= eta * gradient(X, Y_onehot, W, mu)
            step_lst.append(step)
            W_lst.append(W)
            loss_lst.append(loss(X, Y_onehot, W))

        df = pd.DataFrame({
            'step': step_lst,
            'loss': loss_lst
        })
        return df, W
        
    def fit(self):
            self.loss_steps, self.W = gradient_descent(self.X, self.Y)

    def loss_plot(self):
        return self.loss_steps.plot(
            x='step',
            y='loss',
            xlabel='step',
            ylabel='loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)
