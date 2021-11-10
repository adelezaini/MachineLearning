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


