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
from random import random, seed
from regan import create_xyz_dataset, create_X, Split_and_Scale, OLS_solver, MSE, R2, ridge_reg, lasso_reg
import matplotlib.pyplot as plt

savefigure = True

np.random.seed(1234)

#Degree of polynomial
degree=5
# Datapoints (squared root of datapoints -> meshgrid)
n = 25
# Paramaters of noise distribution
mu_N = 0; sigma_N = 0.2

# Create vanilla dataset:
x,y,z = create_xyz_dataset(n,mu_N, sigma_N)

X = create_X(x, y, degree)
X_train, X_test, z_train, z_test = Split_and_Scale(X,np.ravel(z)) #StardardScaler, test_size=0.2, scale=true

ols_beta, ols_model, ols_predict  = OLS_solver(X_train, X_test, z_train, z_test)
print("Training MSE for OLS:", round(MSE(z_train,ols_model),4))
print("Test MSE for OLS", round(MSE(z_test,ols_predict),4))

lambdas = np.logspace(-30,10,num=50)

MSE_lmd_ridge_train = []
MSE_lmd_ridge_test = []
MSE_lmd_lasso_train = []
MSE_lmd_lasso_test = []

for lmd in lambdas:
    _,ridge_model ,ridge_predict  = ridge_reg(X_train, X_test, z_train, z_test, lmd=lmd)
    lasso_model,lasso_predict  = lasso_reg(X_train, X_test, z_train, z_test, lmd=lmd)
    MSE_lmd_ridge_train.append(MSE(ridge_model,z_train))
    MSE_lmd_ridge_test.append(MSE(ridge_predict,z_test))
    MSE_lmd_lasso_train.append(MSE(lasso_model,z_train))
    MSE_lmd_lasso_test.append(MSE(lasso_predict,z_test))


fig = plt.figure(figsize = ( 10, 7))
ax = fig.add_subplot(1,1,1)
ax.plot(lambdas,MSE_lmd_ridge_train, label ="Ridge train", color = 'red', linestyle = 'dashed')  
ax.plot(lambdas,MSE_lmd_ridge_test, label ="Ridge test", color = 'red')
ax.plot(lambdas,MSE_lmd_lasso_train, label ="Lasso train", color = 'green', linestyle = 'dashed')  
ax.plot(lambdas,MSE_lmd_lasso_test, label ="Lasso test", color = 'green')  
ax.set_xscale('log')

plt.xlabel("$\lambda$")
plt.ylabel("MSE")
plt.title(f"Plot of the MSE for different $\lambda$ for complexity {degree}")
plt.legend()
plt.grid()
if savefigure:
    plt.savefig("../reports/Proj1_Plots/Task5/MSE_lambdas.png")
plt.show() 
