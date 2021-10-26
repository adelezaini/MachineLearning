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
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# FrankeFunction: a two-variables function to create the dataset of our vanilla problem
def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4
 
# 3D plot of FrankeFunction
def Plot_FrankeFunction(x,y,z, title="Dataset"):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()
    
# Create xyz dataset from the FrankeFunction with a added normal distributed noise
def create_xyz_dataset(n,mu_N, sigma_N):
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)

    x,y = np.meshgrid(x,y)
    z = FrankeFunction(x,y) +mu_N +sigma_N*np.random.randn(n,n)
    
    return x,y,z

# Error analysis: MSE and R2 score
def R2(z_data, z_model):
    return 1 - np.sum((z_data - z_model) ** 2) / np.sum((z_data - np.mean(z_data)) ** 2)
def MSE(z_data,z_model):
    n = np.size(z_model)
    return np.sum((z_data-z_model)**2)/n

# SVD theorem
def SVD(A):
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
    
# SVD inversion
def SVDinv(A):
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


# Design matrix for two indipendent variables x,y
def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta, number of feutures (degree of polynomial)
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def scale_Xz(X_train, X_test, z_train, z_test, with_std=False):
    scaler_X = StandardScaler(with_std=with_std) #with_std=False
    scaler_X.fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_z = StandardScaler(with_std=with_std) #with_std=False
    z_train = np.squeeze(scaler_z.fit_transform(z_train.reshape(-1, 1))) #scaler_z.fit_transform(z_train) #
    z_test = np.squeeze(scaler_z.transform(z_test.reshape(-1, 1))) #scaler_z.transform(z_test) #  
    return X_train, X_test, z_train, z_test

# Splitting and rescaling data (rescaling is optional)
# Default values: 20% of test data and the scaler is StandardScaler without std.dev.
def Split_and_Scale(X,z,test_size=0.2, scale=True, with_std=False):

    #Splitting training and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size)

    # Rescaling X and z (optional)
    if scale:
        X_train, X_test, z_train, z_test = scale_Xz(X_train, X_test, z_train, z_test, with_std=with_std)
      
    return X_train, X_test, z_train, z_test

# OLS equation
def OLS_solver(X_train, X_test, z_train, z_test):

	# Calculating Beta Ordinary Least Square Equation with matrix pseudoinverse
    # Altervatively to Numpy pseudoinverse it is possible to use the SVD theorem to evalute the inverse of a matrix (even in case it is singular). Just replace 'np.linalg.pinv' with 'SVDinv'.
	ols_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train

	z_tilde = X_train @ ols_beta # z_prediction of the train data
	z_predict = X_test @ ols_beta # z_prediction of the test data
  
	return ols_beta, z_tilde, z_predict


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

# Plot MSE in function of complexity of the model (rolling mean)
def plot_ols_complexity(x, y, z, maxdegree = 20, title="MSE as a function of model complexity"):

    complexity = np.arange(0,maxdegree+1)
    MSE_train_set = []
    MSE_test_set = []

    for degree in complexity:

        X = create_X(x, y, degree)
        X_train, X_test, z_train, z_test = Split_and_Scale(X,np.ravel(z)) #StardardScaler, test_size=0.2, scale=true
        ols_beta, z_tilde,z_predict = OLS_solver(X_train, X_test, z_train, z_test)

        MSE_train_set.append(MSE(z_train,z_tilde))
        MSE_test_set.append(MSE(z_test,z_predict))
    
    plt.figure( figsize = ( 10, 7))
        
    MSE_train_mean, MSE_train_down, MSE_train_up = Rolling_Mean(MSE_train_set)
    plt.plot(complexity, MSE_train_mean, label ="Train (rolling ave.)", color="purple")
    plt.fill_between(complexity, MSE_train_down, MSE_train_up, alpha=0.2, color="purple")
    MSE_test_mean, MSE_test_down, MSE_test_up = Rolling_Mean(MSE_test_set)
    plt.plot(complexity, MSE_test_mean, label ="Test (rolling ave.)", color="orange")
    plt.fill_between(complexity, MSE_test_down, MSE_test_up, alpha=0.2, color="orange")
    
    plt.plot(complexity, MSE_train_set, '--', alpha=0.3, color="purple", label ="Train (actual values)")
    plt.plot(complexity, MSE_test_set, '--', alpha=0.3, color="orange", label ="Test (actual values)")
     
    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    plt.xlim(complexity[~np.isnan(MSE_train_mean)][0]-1,complexity[-1]+1)
    plt.title("Plot of the MSE as a function of complexity of the model\n– Rolling mean and one-sigma region –")
    plt.legend()
    plt.grid()
    plt.show()
    
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


