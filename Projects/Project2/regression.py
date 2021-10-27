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
from sklearn.linear_model import SGDRegressor


# Error analysis
def R2(y_data, y_model):
    """Compute the R2 score of the two given values"""
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    """Compute the Mean Square Error of the two given values"""
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n
    
def train_test_rescale(X_train, X_test, y_train, y_test, with_std=False):
    """Rescale train and test data using StandardScaler().
    The standard deviation correction is not applied by default.
    It's the analogous of train_test_split() by sklearn"""
    
    if len(y_train.shape) > 1:
        print("You forgot to ravel your outputs! \n Automatically ravelled in train_test_rescale()")
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
    
    scaler_X = StandardScaler(with_std=with_std)
    scaler_X.fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler(with_std=with_std)
    y_train = np.squeeze(scaler_y.fit_transform(y_train.reshape(-1, 1))) #scaler_y.fit_transform(y_train) #
    y_test = np.squeeze(scaler_y.transform(y_test.reshape(-1, 1))) #scaler_y.transform(y_test) #
    
    return X_train, X_test, y_train, y_test

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
    
    
def GD(X, y, gradient, eta = 0.1, Niterations = 1000):
    """Gradient Descent Algorithm
    
        Args:
        - X (array): design matrix (training data)
        - y (array): output dataset (training data)
        - gradient (function): function to compute the gradient
        - eta (float): learning rate
        - Niterations (int): number of iteration
        
        Returns:
        beta/theta-values"""
    theta = np.random.randn(X.shape[1],1)

    for iter in range(Niterations):
        gradients = grandient(X, y, theta) #2.0/X.shape[0] * X.T @ ((X @ beta) - y)
        theta -= eta*gradients
        
    return theta
    
    
def learning_schedule(t, t0=5, t1=50):
    return t0/(t+t1)
    
def SGD(X,y, gradient, n_epochs, m, t0=5, t=50):
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
        
    theta = np.random.randn(X.shape[1],1)
    
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            Xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = gradient(Xi, yi, theta) * X.shape[0] #2.0 * Xi.T @ ((Xi @ theta)-yi)
            eta = learning_schedule(epoch*m+i, t0=t0, t1=t1)
            theta = theta - eta*gradients
            
    return theta
    
    
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
        self.intercept = self.solver(X_train, y_train)[0]
        self.mean = np.mean(y_train)
        self.std = np.std(y_train)
        self.y_max = np.max(y_train)
        self.y_min = np.min(y_train)
        
        return self
            
    def rescale(self, with_std=False): #Improvement: pass the Scaler (change descale)
        """ y needs to be raveled """
        self.scale = True
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_rescale(self.X_train, self.X_test, self.y_train, self.y_test, with_std = with_std)
        
        return self
        
    def solver(self, X, y, lmb):
        """Regressione equation"""
        raise NotImplementedError("Method LinearRegression.solver is abstract and cannot be called")
    
    def fit(self):
        """Fit the model and return beta-values"""
        raise NotImplementedError("Method LinearRegression.fit is abstract and cannot be called")
        
    def fitSVD(self):
        """Fit the model and return beta-values, using SVD theorem to evalute the inverse of the matrix"""
        raise NotImplementedError("Method LinearRegression.fitSVD is abstract and cannot be called")
        
    def fit_SK(self):
        """Fit the model and return beta-values, using Scikit-Learn"""
        raise NotImplementedError("Method LinearRegression.fit_SK is abstract and cannot be called")
        
    def fitGD(self):
        """Fit the model and return beta-values, using the Gradient Descent"""
        raise NotImplementedError("Method LinearRegression.fitGD is abstract and cannot be called")
    
    def fitSGD(self):
        """Fit the model and return beta-values, using the Stochastic Gradient Descent"""
        raise NotImplementedError("Method LinearRegression.fitSGD is abstract and cannot be called")
        
    def get_param(self):
        return self.beta
    
    def predict(self, X):
        """Evaluate the prediction
        
        Args:
        X (array): design matrix
        
        Returns X*beta
        """
        return self.X @ self.beta
        
    def predict_train(self):
        return self.predict(self.X_train)
        
    def predict_test(self):
        return self.predict(self.X_test)
        
    def rescaled_predict(self, X): # works with StandardScaler
        beta = self.beta
        beta[0] = self.intercept
        return X @ beta
    
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
        
"""
    def predict(self, X):
        Fit the model and return the prediction
        
        Args:
        X (array): design matrix

        Returns X*beta
        
        raise NotImplementedError("Method LinearRegression.predict is abstract and cannot be called")
"""

class OLSRegression(LinearRegression):
    
    def __init__(self, X, y):
        super().__init__(X, y)
        
    def solver(self, X, y, lmb = 0):
        return np.linalg.pinv(X.T @ X) @ X.T @ y
        
    def fit(self):
        self.beta = self.solver(self.X_train, self.y_train)
        return self.beta
        
    def fitSVD(self):
        self.beta = SVDinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
        return self.beta
        
    def fit_SK(self):
        self.beta = SVDinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
        return self.beta
        
    def gradient(X, y, beta):
        return 2.0/X.shape[0] * X.T @ ((X @ beta) - y) # X.shape[0]=number of input (training) data
            
    def fitGD(self, eta = 0.1, Niterations = 1000):
                
        self.beta = GD(self.X_train, self.y_train, gradient=self.gradient, eta = eta, Niterations = Niterations)
        
        return self.beta
    
    def fitSGD(self, n_epochs, m, t0 = 5, t1 = 50):
          
        self.beta = SGD(X = self.X_train, y = self.y_train, gradient = self.gradient, n_epochs = n_epochs, m = m, t0 = t0, t1 = t1)
        
        return self.beta
        
    def fitSGD_SK(self):
        """
        sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1, fit_intercept=False)
        sgdreg.fit(self.X_train,self.y_train)
        self.beta = sgdreg.coef_
        if self.scale:
            self.beta[0] = sgdreg.intercept_
        
        """
        return self.beta
        
        
class RidgeRegression(LinearRegression):
    
    def __init__(self, X, y, lmb = 1e-12):
        super().__init__(X, y)
        self.lmd = lmd
        
    def set_lambda(self, lmb):
        self.lmb=lmb
        
    def solver(self, X, y, lmd):
        return np.linalg.pinv(X.T @ X + lmd * np.eye(len(self.X.T))) @ X.T @ y
        
    def fit(self):
        self.beta = self.solver(self.X_train, self.y_train, self.lmb)
        return self.beta
        
    def fitSVD(self):
        self.beta = SVDinv(self.X_train.T @ self.X_train + self.lmd * np.eye(len(self.X_train.T))) @ self.X_train.T @ self.y_train
        return self.beta
        
    def fit_SK(self):
        #self.beta = SVDinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
        return self.beta
        
    def gradient(X, y, beta):
        return #2.0/X.shape[0] * X.T @ ((X @ beta) - y) # X.shape[0]=number of input (training) data
            
    def fitGD(self, eta = 0.1, Niterations = 1000):
                
        self.beta = GD(self.X_train, self.y_train, gradient=self.gradient, eta = eta, Niterations = Niterations)
        
        return self.beta
    
    def fitSGD(self, n_epochs, m, t0 = 5, t1 = 50):
          
        self.beta = SGD(X = self.X_train, y = self.y_train, gradient = self.gradient, n_epochs = n_epochs, m = m, t0 = t0, t1 = t1)
        
        return self.beta
        
    def fitSGD_SK(self):
        """
        sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
        sgdreg.fit(x,y.ravel())
        self.beta = np.append(sgdreg.intercept_, sgdreg.coef_).reshape([self.X_train.shape[1],1])
        """
        return self.beta
      

class LassoRegression(LinearRegression):

    def __init__(self, X, y, lmb = 1e-12):
        super().__init__(X, y)
        self.lmd = lmd
        
    def set_lambda(self, lmb):
        self.lmb=lmb
    
    def fit(self):
        RegLasso = linear_model.Lasso(self.lmd)
        self.beta = RegLasso.fit(self.X_train,self.y_train)
        return self.beta
        
    def solver(self, X, y, lmd):
        RegLasso = linear_model.Lasso(lmd)
        return RegLasso.fit(X, y)
        
    def fit(self):
        self.beta = self.solver(self.X_train, self.y_train, self.lmb)
        return self.beta
        
    """
    def fit_SK(self):
        return self.fit()
        
    def fitGD(self):
        return self.fit()
        
    def fitSGD(self):
        return self.fit()
        
    def gradient(X, y, beta):
        return 2.0/X.shape[0] * X.T @ ((X @ beta) - y) # X.shape[0]=number of input (training) data
            
    def fitGD(self, eta = 0.1, Niterations = 1000):
                
        self.beta = GD(self.X_train, self.y_train, gradient=self.gradient, eta = eta, Niterations = Niterations)
        
        return self.beta
    
    def fitSGD(self, n_epochs, m, t0 = 5, t1 = 50):
          
        self.beta = SGD(X = self.X_train, y = self.y_train, gradient = self.gradient, n_epochs = n_epochs, m = m, t0 = t0, t1 = t1)
        
        return self.beta
        
    def fitSGD_SK(self):
    
        sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
        sgdreg.fit(x,y.ravel())
        self.beta = np.append(sgdreg.intercept_, sgdreg.coef_).reshape([self.X_train.shape[1],1])
        
        return self.beta
    """
    
    
    
"""
def ols_reg(X_train, X_test, y_train, y_test):

	# Calculating Beta Ordinary Least Square Equation with matrix pseudoinverse
    # Altervatively to Numpy pseudoinverse it is possible to use the SVD theorem to evalute the inverse of a matrix (even in case it is singular). Just replace 'np.linalg.pinv' with 'SVDinv'.
	ols_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

	y_tilde = X_train @ ols_beta # y_prediction of the train data
	y_predict = X_test @ ols_beta # y_prediction of the test data
  
	return ols_beta, y_tilde, y_predict
 
 def ridge_reg(X_train, X_test, y_train, y_test, lmd = 10**(-12)):
 
    ridge_beta = np.linalg.pinv(X_train.T @ X_train + lmd*np.eye(len(X_train.T))) @ X_train.T @ y_train #psudoinverse
    y_model = X_train @ ridge_beta #calculates model
    y_predict = X_test @ ridge_beta

    #finds the lambda that gave the best MSE
    #best_lamda = lambdas[np.where(MSE_values == np.min(MSE_values))[0]]

    return ridge_beta, y_model, y_predict
    
def lasso_reg(X_train, X_test, y_train, y_test, lmd = 10**(-12)):

    RegLasso = linear_model.Lasso(lmd)
    _ = RegLasso.fit(X_train,y_train)
    y_model = RegLasso.predict(X_train)
    y_predict = RegLasso.predict(X_test)

    return y_model, y_predict
"""

# Return the rolling mean of a vector and two values at one sigma from the rolling average
def Rolling_Mean(vector, windows=3):
    vector_df = pd.DataFrame({'vector': vector})
    # computing the rolling average
    rolling_mean = vector_df.vector.rolling(windows).mean().to_numpy()
    # computing the values at two sigmas from the rolling average
    rolling_std = vector_df.vector.rolling(windows).std().to_numpy()
    value_up = rolling_mean + rolling_std
    value_down = rolling_mean - rolling_std
    
    return rolling_mean, value_down, value_up
    


