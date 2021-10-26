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
def R2(z_data, z_model):
    """Compute the R2 score of the two given values"""
    return 1 - np.sum((z_data - z_model) ** 2) / np.sum((z_data - np.mean(z_data)) ** 2)
def MSE(z_data,z_model):
    """Compute the Mean Square Error of the two given values"""
    n = np.size(z_model)
    return np.sum((z_data-z_model)**2)/n

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
    
    
def learning_schedule(t, t0=5, t1=50):
    return t0/(t+t1)
    
    
class LinearRegression:
    """A class that gathers OLS, Ridge, Lasso methods

    The 'fit' method needs to be implemented."""

    def __init__(self, X, z):
        self.X = X
        self.z = z
        
    def split(self,test_size=0.2):
        self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(self.X, self.z, test_size=test_size)
        return self.X_train, self.X_test, self.z_train, self.z_test
            
    def rescale(self, with_std=False): #Improvement: pass the Scaler
        """ z needs to be raveled """
        scaler_X = StandardScaler(with_std=with_std)
        scaler_X.fit(self.X_train)
        self.X_train = scaler_X.transform(self.X_train)
        self.X_test = scaler_X.transform(self.X_test)

        scaler_z = StandardScaler(with_std=with_std)
        self.z_train = np.squeeze(scaler_z.fit_transform(self.z_train.reshape(-1, 1))) #scaler_z.fit_transform(z_train) #
        self.z_test = np.squeeze(scaler_z.transform(self.z_test.reshape(-1, 1))) #scaler_z.transform(z_test) #
        
        return self.X_train, self.X_test, self.z_train, self.z_test
    
    def fit(self):
        """Fit the model and return beta-values"""
        raise NotImplementedError("Method LinearRegression.fit is abstract and cannot be called")
        
    def fitSVD(self):
        """Fit the model and return beta-values, using SVD theorem to evalute the inverse of the matrix"""
        raise NotImplementedError("Method LinearRegression.fitSVD is abstract and cannot be called")
        
    def fitSGD(self):
        """Fit the model and return beta-values, using the Stochastic Gradient Descent"""
        raise NotImplementedError("Method LinearRegression.fitSGD is abstract and cannot be called")
    
    def predict_train(self):
        return self.X_train @ self.beta
        
    def predict_test(self):
        return self.X_test @ self.beta
    
    def MSE_train(self, prec=4):
        return np.round(MSE(self.z_train,self.predict_train()),prec)
        
    def MSE_test(self, prec=4):
        return np.round(MSE(self.z_test,self.predict_test()),prec)
        
    def R2_train(self, prec=4):
        return np.round(R2(self.z_train,self.predict_train()),prec)
        
    def R2_test(self, prec=4):
        return np.round(R2(self.z_test,self.predict_test()),prec)
        
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
    
    def __init__(self, X, z):
        super().__init__(X, z)
        
    def split(self, test_size=0.2):
        return super().split(test_size=test_size)
        
    def rescale(self, with_std=False):
        return super().rescale(with_std=with_std)
        
    def fit(self):
        self.beta = np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.z_train
        return self.beta
        
    def fitSVD(self):
        self.beta = SVDinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.z_train
        return self.beta
        
    def fitGD(self, n, eta = 0.1, Niterations = 1000):
    
        theta = np.random.randn(self.X.shape[1],1)

        for iter in range(Niterations):
            gradients = 2.0/n * self.X_train.T @ ((self.X_train @ theta) - self.z_train)
            theta -= eta*gradients
                
        self.beta = theta
        
        return self.beta
        
    def fitSGD(self, n_epochs, m, t0=5, t1=50):
    
        theta = np.random.randn(self.X.shape[1],1)
        
        for epoch in range(n_epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                Xi = self.X_train[random_index:random_index+1]
                zi = self.z_train[random_index:random_index+1]
                gradients = 2.0 * Xi.T @ ((Xi @ theta)-zi)
                eta = learning_schedule(epoch*m+i, t0=t0, t1=t1)
                theta = theta - eta*gradients
                
        self.beta = theta
        
        return self.beta
        
    def fitSGD_SK:
        sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
        sgdreg.fit(x,y.ravel())
        self.beta = np.append(sgdreg.intercept_, sgdreg.coef_).reshape([self.X_train.shape[1],1])
        
        return self.beta
          
    def predict_train(self):
        return super().predict_train()
        
    def predict_test(self):
        return super().predict_test()
        
    def MSE_train(self, prec=4):
        return super().MSE_train(prec=prec)
        
    def MSE_test(self, prec=4):
        return super().MSE_test(prec=prec)
        
    def R2_train(self, prec=4):
        return super().R2_train(prec=prec)
        
    def R2_test(self, prec=4):
        return super().R2_test(prec=prec)
        
    def Confidence_Interval(self, sigma=1):
        return super().Confidence_Interval(sigma=sigma)
        
class RidgeRegression(LinearRegression):
    
    def __init__(self, X, z, lmb = 1e-12):
        super().__init__(X, z)
        self.lmd = lmd
        
    def split(self, test_size=0.2):
        return super().split(test_size=test_size)
        
    def rescale(self, with_std=False):
        return super().rescale(with_std=with_std)
    
    def fit(self):
        self.beta = np.linalg.pinv(self.X_train.T @ self.X_train + self.lmd * np.eye(len(self.X_train.T))) @ self.X_train.T @ self.z_train
        return self.beta
        
    def fitSVD(self):
        self.beta = SVDinv(self.X_train.T @ self.X_train + self.lmd * np.eye(len(self.X_train.T))) @ self.X_train.T @ self.z_train
        return self.beta
          
    def predict_train():
        return super().predict_train()
        
    def predict_test(self):
        return super().predict_test()

    def MSE_train(self, prec=4):
        return super().MSE_train(prec=prec)
        
    def MSE_test(self, prec=4):
        return super().MSE_test(prec=prec)
        
    def R2_train(self, prec=4):
        return super().R2_train(prec=prec)
        
    def R2_test(self, prec=4):
        return super().R2_test(prec=prec)
        
    def Confidence_Interval(self, sigma=1):
        return super().Confidence_Interval(sigma=sigma)

class LassoRegression(LinearRegression):

    def __init__(self, X, z, lmb = 1e-12):
        super().__init__(X, z)
        self.lmd = lmd
        
    def split(self, test_size=0.2):
        return super().split(test_size=test_size)
        
    def rescale(self, with_std=False):
        return super().rescale(with_std=with_std)
    
    def fit(self):
        RegLasso = linear_model.Lasso(self.lmd)
        self.beta = RegLasso.fit(self.X_train,self.z_train)
        return self.beta
        
    def fit(self):
        return fit()
          
    def predict_train(self):
        return super().predict_train()
        
    def predict_test(self):
        return super().predict_test()
        
    def MSE_train(self, prec=4):
        return super().MSE_train(prec=prec)
        
    def MSE_test(self, prec=4):
        return super().MSE_test(prec=prec)
        
    def R2_train(self, prec=4):
        return super().R2_train(prec=prec)
        
    def R2_test(self, prec=4):
        return super().R2_test(prec=prec)
        
    def Confidence_Interval(self, sigma=1):
        return super().Confidence_Interval(sigma=sigma)
    
"""
def ols_reg(X_train, X_test, z_train, z_test):

	# Calculating Beta Ordinary Least Square Equation with matrix pseudoinverse
    # Altervatively to Numpy pseudoinverse it is possible to use the SVD theorem to evalute the inverse of a matrix (even in case it is singular). Just replace 'np.linalg.pinv' with 'SVDinv'.
	ols_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train

	z_tilde = X_train @ ols_beta # z_prediction of the train data
	z_predict = X_test @ ols_beta # z_prediction of the test data
  
	return ols_beta, z_tilde, z_predict
 
 def ridge_reg(X_train, X_test, z_train, z_test, lmd = 10**(-12)):
 
    ridge_beta = np.linalg.pinv(X_train.T @ X_train + lmd*np.eye(len(X_train.T))) @ X_train.T @ z_train #psudoinverse
    z_model = X_train @ ridge_beta #calculates model
    z_predict = X_test @ ridge_beta

    #finds the lambda that gave the best MSE
    #best_lamda = lambdas[np.where(MSE_values == np.min(MSE_values))[0]]

    return ridge_beta, z_model, z_predict
    
def lasso_reg(X_train, X_test, z_train, z_test, lmd = 10**(-12)):

    RegLasso = linear_model.Lasso(lmd)
    _ = RegLasso.fit(X_train,z_train)
    z_model = RegLasso.predict(X_train)
    z_predict = RegLasso.predict(X_test)

    return z_model, z_predict
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
    


