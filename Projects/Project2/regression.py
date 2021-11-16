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
from activations import *
    
def MSE_BS(y_pred_matrix, y_data):
    return np.mean( np.mean((y_data.reshape(-1,1) - y_pred_matrix)**2, axis=1, keepdims=True) )
        
class LinearRegression:
    """A class that gathers OLS, Ridge, Lasso methods

    The 'fit' method needs to be implemented."""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        self.intercept = self.solver(X, y)[0]
        self.mean = np.mean(y)
        self.std = np.std(y)
        self.y_max = np.max(y)
        self.y_min = np.min(y)
        self.scale = False
        
    def split(self,test_size=0.2):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)
        
        # set parameters automaticaly after splitting
        self.intercept = self.solver(self.X_train, self.y_train)[0]
        self.mean = np.mean(self.y_train)
        self.std = np.std(self.y_train)
        self.y_max = np.max(self.y_train)
        self.y_min = np.min(self.y_train)
        
        return self
            
    def rescale(self, with_std=False): #Improvement: pass the Scaler (change descale)
        """ y needs to be raveled """
        self.scale = True
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_rescale(self.X_train, self.X_test, self.y_train, self.y_test, with_std = with_std)
        
        return self
        
    def solver(self, X, y, lmd, svd = False):
        """Regressione equation"""
        raise NotImplementedError("Method LinearRegression.solver is abstract and cannot be called")
        
    def gradient(self, X, y, beta, lmd):
        """Gradient equation"""
        raise NotImplementedError("Method LinearRegression.gradient is abstract and cannot be called")
    
    def fit(self):
        """Fit the model and return beta-values"""
        self.beta = self.solver(self.X_train, self.y_train, self.lmd)
        return self.beta
    
    def fit_SK(self): # to be tested
        """Fit the model and return beta-values, using Scikit-Learn"""
        raise NotImplementedError("Method LinearRegression.fit_SK is abstract and cannot be called")
        
    def fitGD(self, eta = 0.1, Niterations = 1000):
        """Fit the model and return beta-values, using the Gradient Descent"""
        self.beta = GD(self.X_train, self.y_train, lmd=self.lmd, gradient=self.gradient, eta = eta, Niterations = Niterations)
        return self.beta
    
    def fitSGD(self, n_epochs, M, opt = "SGD", eta0 = 0.1, eta_type = 'schedule', t0=5, t1=50, momentum = 0., rho = 0.9, b1 = 0.9, b2 = 0.999):
        """Fit the model and return beta-values, using the Stochastic Gradient Descent.
        Description of the various paramentes in doc of SGD(). """
        self.beta = SGD(X = self.X_train, y = self.y_train, lmd=self.lmd, gradient = self.gradient, n_epochs = n_epochs, M = M, opt = opt, eta0 = eta0, eta_type = eta_type, t0 = t0, t1 = t1, momentum = momentum, rho = rho, b1 = b1, b2 = b2)
        return self.beta
        
    def predictSGD_BS(self, n_epochs, M, opt = "SGD", eta0 = 0.1, eta_type = 'schedule', t0 = 5, t1 = 50, n_boostraps=100, momentum = 0., rho = 0.9, b1 = 0.9, b2 = 0.999):
        """Predict y
        The Stochastic Gradient Descent and Bootstrap esampling algorithm are implemented
        
        Returns:
        (m x n_bootstraps) matrix with the column vectors y_pred for each bootstrap iteration.
        """
        y_pred_boot = np.empty((self.y_test.shape[0], n_boostraps))
        for i in range(n_boostraps):
            # Draw a sample of our dataset
            X_sample, y_sample = resample(self.X_train, self.y_train)
            # Perform OLS equation
            beta = SGD(X = X_sample, y = y_sample, lmd=self.lmd, gradient = self.gradient, n_epochs = n_epochs, M = M, opt = opt, eta0 = eta0, eta_type = eta_type, t0 = t0, t1 = t1, momentum = momentum, rho = rho, b1 = b1, b2 = b2)
            y_pred = self.X_test @ beta
            y_pred_boot[:, i] = y_pred.ravel()

        return y_pred_boot
    
    def get_param(self):
        """Return beta-values"""
        return self.beta
    
    def predict(self, X):
        """Evaluate the prediction
        
        Args:
        X (array): design matrix
        
        Returns X*beta
        """
        return X @ self.beta
        
    def predict_train(self):
        return self.predict(self.X_train)
        
    def predict_test(self):
        return self.predict(self.X_test)
        
    def rescaled_predict(self, X): # works with StandardScaler
        beta = self.beta
        beta[0] = self.intercept
        return X @ beta
        
    def print(self,q):
        print(q)
    
    def MSE_train(self, prec=4):
        return np.round(MSE(self.y_train,self.predict_train()),prec)
        
    def MSE_test(self, prec=4):
        return np.round(MSE(self.y_test,self.predict_test()),prec)
        
    def R2_train(self, prec=4):
        return np.round(R2(self.y_train,self.predict_train()),prec)
        
    def R2_test(self, prec=4):
        return np.round(R2(self.y_test,self.predict_test()),prec)
        
    def Confidence_Interval(self, sigma=1):
        #Calculates variance of beta, extracting just the diagonal elements of the matrix
        #var(B_j)=sigma^2*(X^T*X)^{-1}_{jj}
        beta_variance = np.diag(sigma**2 * np.linalg.pinv(self.X.T @ self.X))
        ci1 = self.beta - 1.96 * np.sqrt(beta_variance)/(self.X.shape[0])
        ci2 = self.beta + 1.96 * np.sqrt(beta_variance)/(self.X.shape[0])
        print('Confidence interval of β-estimator at 95 %:')
        ci_df = {r'$β_{-}$': ci1,
                 r'$β_{ols}$': self.beta,
                 r'$β_{+}$': ci2}
        ci_df = pd.DataFrame(ci_df)
        display(np.round(ci_df,3))
        return ci1, ci2
        

class OLSRegression(LinearRegression):
    
    def __init__(self, X, y):
        self.lmd = 0
        super().__init__(X, y)
        
    def solver(self, X, y, lmd = 0, svd = False):
        if not svd:
            return np.linalg.pinv(X.T @ X) @ X.T @ y
        else:
            return SVDinv(X.T @ X) @ X.T @ y
        
    def gradient(self, X, y, beta, lmd=0):
        return 2.0/np.shape(X)[0] * X.T @ ((X @ beta) - y)  # X.shape[0]=number of input (training) data
        
    def fit_SK(self):
        degree = np.shape(self.X_train)[1] - 1
        model = make_pipeline(PolynomialFeatures(degree = degree), sk.LinearRegression(fit_intercept=False))
        self.beta = model.fit(self.X_train, self.y_train)
        return self.beta
        
    def fitSGD_SK(self, eta0=0.1, t0 = 5, t1 = 50, max_iter = 500, penalty=None): # parameters need to step them as the other methods as default !
        """Stochastic Gredient Descent provided by Scikit Learn"""
        alpha = 1. / t0
        # t0 = t1 (saw that are equivalent in the documentation of sklearn.SGDRegressor, but no way to set it form the outside)
        sgdreg = sk.SGDRegressor(eta0=eta0, alpha = alpha, max_iter = max_iter, penalty = penalty, fit_intercept=False)
        sgdreg.fit(self.X_train,self.y_train)
        self.beta = sgdreg.coef_
        if self.scale:
            self.beta[0] = sgdreg.intercept_
        return self.beta
        
        
class RidgeRegression(LinearRegression):
    
    def __init__(self, X, y, lmd = 1e-12):
        super().__init__(X, y)
        self.lmd = lmd
        
    def set_lambda(self, lmd):
        self.lmd=lmd
        
    def solver(self, X, y, lmd = 0, svd = False):
        if not svd:
            return np.linalg.pinv(X.T @ X + lmd * np.eye(len(X.T))) @ X.T @ y
        else:
            return SVDinv(X.T @ X + lmd * np.eye(len(self.X.T))) @ X.T @ y
            
    def gradient(self, X, y, beta, lmd=0):
        return 2.0/np.shape(X)[0] * X.T @ ((X @ beta) - y) - 2. * lmd * beta  # X.shape[0]=number of input (training) data
        
    def fit_SK(self):
        Ridge = sk.Ridge(self.lmb,fit_intercept=False)
        Ridge.fit(self.X_train, self.y_train)
        self.beta = Ridge.coef_
        if self.scale:
            self.beta[0] = Ridge.intercept_
        return self.beta
        
    def fitSGD_SK(self):
        pass
      

class LassoRegression(LinearRegression):  # to be tested

    def __init__(self, X, y, lmd = 1e-12):
        super().__init__(X, y)
        self.lmd = lmd
        
    def set_lambda(self, lmd):
        self.lmd=lmd
        
    def solver(self, X, y, lmd = 0, svd = False):
        self.Lasso = sk.Lasso(lmd)
        self.beta = self.Lasso.coef_
        if self.scale:
            self.beta[0] = self.Lasso.intercept_
        return self.beta
        
    def gradient(self, X, y, beta, lmd):
        pass
        
    def predict(self, X):
        return self.Lasso.predict(X)
        
        
        

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
        '''
        Predict whether the label is 0 or 1
        
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
          - y : n_labels of all datapoints, shape (n_labels,)
    """
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
          - y : n_labels of all datapoints, shape (n_labels,)
    """
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
          - y : n_labels of all datapoints, shape (n_labels,)
    """
        self.X = X
        self.y = y

    def loss(X, Y, W):
        """
        Y: onehot encoded
        """
        Z = - X @ W
        N = X.shape[0]
        loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def gradient(X, Y, W, mu):
        """
        Y: onehot encoded
        """
        Z = - X @ W
        P = softmax(Z, axis=1)
        N = X.shape[0]
        gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W
        return gd

    def gradient_descent(X, Y, max_iter=1000, eta=0.1, mu=0.01):
        """
        Very basic gradient descent algorithm with fixed eta and mu
        """
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
