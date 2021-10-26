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

from math import trunc
import numpy as np

from numpy.core.numeric import True_
from regan import *
from imageio import imread
import matplotlib.pyplot as plt

def train_n(n,test_size):
    return int(n*n*(1-test_size))
    
def test_n(n,test_size):
    return int(n*n*test_size)
    
savefigures = True
np.random.seed(1234)

# Load the terrain
terrain = imread('../reports/Proj1_DataFiles/SRTM_data_Norway_2.tif')
N = 30
n = N

z = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(z)[0])
y = np.linspace(0,1, np.shape(z)[1])
x,y = np.meshgrid(x,y)

maxdegree = 35
degree=5
test_size = 0.2

#----- Task1

print("––––––––– TASK 1 –––––––––––––––")
# OLS
X = create_X(x, y, degree)
X_train, X_test, z_train, z_test = Split_and_Scale(X,np.ravel(z)) #StardardScaler, test_size=0.2, scale=true
ols_beta, z_tilde,z_predict = OLS_solver(X_train, X_test, z_train, z_test)

prec=4
print("––––––––––––––––––––––––––––––––––––––––––––")
print("Train MSE for OLS:", np.round(MSE(z_train,z_tilde),prec))
print("Test MSE for OLS:", np.round(MSE(z_test,z_predict),prec))
print("––––––––––––––––––––––––––––––––––––––––––––")
print("Train R2 for OLS:", np.round(R2(z_train,z_tilde),prec))
print("Test R2 for OLS:", np.round(R2(z_test,z_predict),prec))
print("––––––––––––––––––––––––––––––––––––––––––––")


# Confidence interval
beta1, beta2 = Confidence_Interval(ols_beta, X_train)
print("––––––––––––––––––––––––––––––––––––––––––––")

#----- Task2
print("––––––––– TASK 2 –––––––––––––––")
plot_ols_complexity(x,y,z, maxdegree)
print("Train datapoints:", train_n(n,test_size))
print("Test datapoints:", test_n(n,test_size))
bias_variance_complexity(x, y, z.ravel(), maxdegree=15, test_size=test_size)

#----- Task3
print("––––––––– TASK 3 –––––––––––––––")
complexity = []
MSE_train_set_k5 = []
MSE_test_set_k5 = []
MSE_train_set_k10 = []
MSE_test_set_k10 = []

for i in range(0,maxdegree): #goes out of range for high i?


  #OLS no CV
  X = create_X(x, y, i)
  X_train, X_test, z_train, z_test = Split_and_Scale(X,np.ravel(z)) #StardardScaler, test_size=0.2, scale=true
  ols_beta, z_tilde,z_predict = OLS_solver(X_train, X_test, z_train, z_test)

  complexity.append(i)
  #k = 5
  MSE_train, MSE_test = cross_validation(5,X,z.ravel(),solver="OLS")
  MSE_train_set_k5.append(MSE_train)
  MSE_test_set_k5.append(MSE_test)

  #k = 10
  MSE_train, MSE_test = cross_validation(10,X,z.ravel(),solver="OLS")
  MSE_train_set_k10.append(MSE_train)
  MSE_test_set_k10.append(MSE_test)


fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1,1,1)
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
if savefigures:
    plt.savefig("../reports/Proj1_Plots/Task6/CV.png")
plt.show()


print("––––––––– TASK 4-5 –––––––––––––––")
#----------------------------------------------------
#Get MSE vs lambdas

lambdas = np.logspace(-30,20,num=50)

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


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(lambdas,MSE_lmd_ridge_train, label ="Ridge train", color = 'red', linestyle = 'dashed')
ax.plot(lambdas,MSE_lmd_ridge_test, label ="Ridge test", color = 'red')
ax.plot(lambdas,MSE_lmd_lasso_train, label ="Lasso train", color = 'green', linestyle = 'dashed')
ax.plot(lambdas,MSE_lmd_lasso_test, label ="Lasso test", color = 'green')
ax.set_xscale('log')

plt.xlabel("$\lambda$")
plt.ylabel("MSE")
plt.title(f"Plot of the MSE for different $\lambda$ with complexity {degree}")
plt.legend()
plt.grid()
if savefigures:
    plt.savefig("../reports/Proj1_Plots/Task6/MSE_lambdas.png")
plt.show()

#---------------------------------------------------
#RUN cv and bootstrap

#lambdas = [10**x for x in [-12, -6, -3, 0, 3]]

foldername = 'Task6'

run_plot_compare(z, 100, N=n, k=5,poly_degree = 18,plot=True,saveplots=savefigures, foldername = foldername, title="Portion of analyzed terrain")

"""
compare_lmd_BS(z, n, lambdas, maxdegree, solver = 'RIDGE', n_resampling = 100, saveplots = savefigures, foldername = foldername)
compare_lmd_CV(z, n, 5, lambdas, maxdegree, solver = 'RIDGE', saveplots = savefigures, foldername = foldername)
compare_lmd_CV(z, n, 10, lambdas, maxdegree, solver = 'RIDGE', saveplots = savefigures, foldername = foldername)

compare_lmd_BS(z, n, lambdas, maxdegree, solver = 'LASSO', n_resampling = 100, saveplots = savefigures, foldername = foldername)
compare_lmd_CV(z, n, 5, lambdas, maxdegree, solver = 'LASSO', saveplots = savefigures, foldername = foldername)
compare_lmd_CV(z, n, 5, lambdas, maxdegree, solver = 'LASSO', saveplots = savefigures, foldername = foldername)"""


