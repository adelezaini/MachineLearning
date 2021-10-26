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

from random import random, seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regan import create_X, create_xyz_dataset, cross_validation, Split_and_Scale, OLS_solver, MSE

savefigure = True

np.random.seed(1234)

# Datapoints (squared root of datapoints -> meshgrid)
n = 30
# Paramaters of noise distribution
mu_N = 0; sigma_N = 0.1
# Parameter of splitting data
test_size = 0.2
#degree of polynomial
degree = 12
# Create vanilla dataset:
x,y,z = create_xyz_dataset(n,mu_N, sigma_N); z = z.ravel()


# Studying the MSE_train and MSE_test VS complexity with cross validation method
complexity = []
MSE_train_set_k5 = []
MSE_test_set_k5 = []
MSE_train_set_k10 = []
MSE_test_set_k10 = []


MSE_train_noCV = []
MSE_test_noCV = []

for i in range(0,degree): #goes out of range for high i?


	#OLS no CV
	X = create_X(x, y, i)
	X_train, X_test, z_train, z_test = Split_and_Scale(X,np.ravel(z)) #StardardScaler, test_size=0.2, scale=true
	ols_beta, z_tilde,z_predict = OLS_solver(X_train, X_test, z_train, z_test)
	MSE_train_noCV.append(MSE(z_train,z_tilde))
	MSE_test_noCV.append(MSE(z_test,z_predict))

	complexity.append(i)
	#k = 5
	MSE_train, MSE_test = cross_validation(5,X,z,solver="OLS")
	MSE_train_set_k5.append(MSE_train)
	MSE_test_set_k5.append(MSE_test)

	#k = 10
	MSE_train, MSE_test = cross_validation(10,X,z,solver="OLS")
	MSE_train_set_k10.append(MSE_train)
	MSE_test_set_k10.append(MSE_test)


fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1,1,1)
#ax.plot(complexity,MSE_train_noCV, label ="train k=1", color = 'red', linestyle = 'dashed')
#ax.plot(complexity,MSE_test_noCV, label ="test k=1", color = 'red')
ax.plot(complexity,MSE_train_set_k5, label ="train k=5", color = 'green', linestyle = 'dashed')  
ax.plot(complexity,MSE_test_set_k5, label ="test k=5", color = 'green')  
ax.plot(complexity,MSE_train_set_k10, label ="train k=10", color = 'blue', linestyle = 'dashed')  
ax.plot(complexity,MSE_test_set_k10, label ="test k=10", color = 'blue')    
ax.set_yscale('log')

plt.xlabel("complexity")
plt.ylabel("MSE")
plt.title("Plot of the MSE for different number of folds in crossvalidation")
plt.legend()
plt.grid()
if savefigure:
	plt.savefig("../reports/Proj1_Plots/Task3/MSE.png")
plt.show() 

