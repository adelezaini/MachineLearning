# The MIT License (MIT)
#
# Copyright © 2021 Fridtjof Gjengset, Adele Zaini, Gaute Holen
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
from .regression import resample, OLS_solver, ridge_reg, lasso_reg, create_X, Split_and_Scale, Rolling_Mean
import matplotlib.pyplot as plt
import pandas as pd


def Confidence_Interval(beta, X, sigma=1):
    #Calculates variance of beta, extracting just the diagonal elements of the matrix
    #var(B_j)=sigma^2*(X^T*X)^{-1}_{jj}
    beta_variance = np.diag(sigma**2 * np.linalg.pinv(X.T @ X))
    ci1 = beta - 1.96 * np.sqrt(beta_variance)/(X.shape[0])
    ci2 = beta + 1.96 * np.sqrt(beta_variance)/(X.shape[0])
    print('Confidence interval of β-estimator at 95 %:')
    ci_df = {r'$β_{-}$': ci1,
             r'$β_{ols}$': beta,
             r'$β_{+}$': ci2}
    ci_df = pd.DataFrame(ci_df)
    display(np.round(ci_df,3))
    return ci1, ci2

# Bootstrap resampling
# Return a (m x n_bootstraps) matrix with the column vectors z_pred for each bootstrap iteration.
def bootstrap(X_train, X_test, z_train, z_test, n_boostraps=100, solver = "OLS", lmd = 10**(-12)):
    if solver not in ["OLS", "RIDGE", "LASSO"]:
        raise ValueError("solver must be OLS, RIDGE OR LASSO")

    z_pred_boot = np.empty((z_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        # Draw a sample of our dataset
        X_sample, z_sample = resample(X_train, z_train)
        # Perform OLS equation
        if solver == "OLS":
            ols_beta, z_tilde, z_pred = OLS_solver(X_sample, X_test, z_sample, z_test)
        elif solver == "RIDGE":
            ridge_beta, z_tilde, z_pred = ridge_reg(X_sample, X_test, z_sample, z_test, lmd=lmd)
        elif solver == "LASSO":
            z_tilde, z_pred = lasso_reg(X_sample, X_test, z_sample, z_test, lmd=lmd)

        z_pred_boot[:, i] = z_pred.ravel()

    return z_pred_boot
    
# Bias-variance tradeoff

# Note: Expectations and variances taken w.r.t. different training
# data sets, hence the axis=1. Subsequent means are taken across the test data
# set in order to obtain a total value, but before this we have error/bias/variance
# calculated per data point in the test set.
# Note 2: The use of keepdims=True is important in the calculation of bias as this
# maintains the column vector form. Dropping this yields very unexpected results.

# conclude with cross validation

def bias_variance_analysis(X_train, X_test, z_train, z_test, resampling="bootstrap", n_resampling = 100, solver = "OLS", lmd = 10**(-12)):
    if(resampling=="bootstrap"):
        z_pred = bootstrap(X_train, X_test, z_train, z_test, n_boostraps = n_resampling, solver = solver,lmd = lmd)

    """ else:
        z_pred = crossvalidation(X_train, X_test, z_train, z_test, n_resampling)
    """

    error = np.mean( np.mean((z_test.reshape(-1,1) - z_pred)**2, axis=1, keepdims=True) ) # MSE
    bias2 = np.mean( (z_test.reshape(-1,1) - np.mean(z_pred, axis=1, keepdims=True))**2 ) # bias^2
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )

    return error, bias2, variance
    
# Plot bias-variance tradeoff in function of complexity of the model
def bias_variance_complexity(x, y, z, maxdegree=20, n_resampling = 100, test_size = 0.2, plot=True, title="Bias-variance analysis: MSE as a function of model complexity", solver = "OLS", lmd=10**(-12)):

    complexity = np.arange(0,maxdegree+1)
    error = np.zeros(complexity.size)
    bias = np.zeros(complexity.size)
    variance = np.zeros(complexity.size)

    for degree in complexity:
        X = create_X(x, y, degree)
        X_train, X_test, z_train, z_test = Split_and_Scale(X,z,test_size=test_size) #StardardScaler, test_size=0.2, scale=true
        error[degree], bias[degree], variance[degree] = bias_variance_analysis(X_train, X_test, z_train, z_test, n_resampling = n_resampling,solver = solver,lmd = lmd)
    
        # For debugging
        #print('Error:', error[degree])
        #print('Bias^2:', bias[degree])
        #print('Var:', variance[degree])
        #print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

        # test: close to the precision...
        
    if (plot==True):
        plt.figure( figsize = ( 10, 7))
            
        error_mean, error_down, error_up = Rolling_Mean(error,2)
        plt.plot(complexity, error_mean, label ="Error (rolling ave.)", color="purple")
        plt.fill_between(complexity, error_down, error_up, alpha=0.1, color="purple")
        bias_mean, bias_down, bias_up = Rolling_Mean(bias,2)
        plt.plot(complexity, bias_mean, label =r"Bias$^2$ (rolling ave.)", color="forestgreen")
        plt.fill_between(complexity, bias_down, bias_up, alpha=0.1, color="forestgreen")
        variance_mean, variance_down, variance_up = Rolling_Mean(variance,2)
        plt.plot(complexity, variance_mean, label ="Variance (rolling ave.)", color="darkorange")
        plt.fill_between(complexity, variance_down, variance_up, alpha=0.1, color="darkorange")
        
        plt.plot(complexity, error, '--', alpha=0.3, color="purple", label ="Error (actual values)")
        plt.plot(complexity, bias, '--', alpha=0.3, color="forestgreen", label ="Bias (actual values)")
        plt.plot(complexity, variance, '--', alpha=0.3, color="darkorange", label ="Variance (actual values)")
         
        plt.xlim(complexity[~np.isnan(error_mean)][0]-1,complexity[-1]+1)
        title=title+str("\n– Rolling mean and one-sigma region –")
        plt.grid()
    
        """
        plt.plot(complexity, error, label='Error')
        plt.plot(complexity, bias, label=r'$Bias^2$')
        plt.plot(complexity, variance, label='Variance')
        """
        plt.xlabel("Complexity")
        plt.title(title)
        plt.legend()
        plt.show()
    
    return error, bias, variance #its bias2
