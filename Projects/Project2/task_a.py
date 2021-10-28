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
from dataset import create_xyz_dataset, create_X, Plot_FrankeFunction
from regression import OLSRegression, RidgeRegression
import matplotlib.pyplot as plt


np.random.seed(1234)

# Degree of the polynomial
degree=5
# Datapoints (squared root of datapoints -> meshgrid)
n = 25
# Paramaters of noise distribution
mu_n = 0; sigma_n = 0.1

# Create vanilla dataset:
x,y,z = create_xyz_dataset(n,mu_n, sigma_n)
Plot_FrankeFunction(x,y,z, title="Original dataset: : \nFranke Function with stochastic noise")
# Convertion because of meshgrid
z = z.ravel(); n=n**2
X = create_X(x, y, degree)


n_epochs = 50
M = 5   #size of each minibatch
m = int(n/M) #number of minibatches

# Default value
ols = OLSRegression(X,z)
ridge = RidgeRegression(X,z)
ols.split().rescale().fitSGD(n_epochs = n_epochs, m = m)
ridge.split().rescale().fitSGD(n_epochs = n_epochs, m = m)
print("Performance with default values: t0, t1 = 5, 50")
print("––––––––––––––––––––––––––––––––––––––––––––")
print("OLS - MSE:", ols.MSE_test())
print("OLS - R2:", ols.R2_test())
print("Ridge - MSE:", ols.MSE_test())
print("Ridge - R2:", ols.R2_test())
print("––––––––––––––––––––––––––––––––––––––––––––")

t0_vals = [0.05, 0.1, 0.5, 1, 5, 10, 25]
t1_vals = [1, 3, 12, 26, 50, 100, 120]

mse_ols = np.zeros((len(t0_vals), len(t1_vals)))
mse_ridge = np.zeros((len(t0_vals), len(t1_vals)))

for i in range(len(t0_vals)):
    for j in range(len(t1_vals)):
        
        ols.fitSGD(n_epochs, m, t0_vals[i], t1_vals[j])
        ridge.fitSGD(n_epochs, m, t0_vals[i], t1_vals[j])
        
        mse_ols[i][j] = ols.MSE_test()
        mse_ridge[i][j] = ridge.MSE_test()
        
        if mse_ols[i][j] > 1.:
            mse_ols[i][j] = 1.
            
        if mse_ridge[i][j] > 1.:
            mse_ridge[i][j] = 1.
    
import seaborn as sns

sns.set()

def find_min_indexes(A):
    return np.array(np.where(A == A.min())).flatten()

ind= find_min_indexes(mse_ols)
print(r"OLS: Best learning rate $\eta_0$ = "+str(t0_vals[ind[0]]/t1_vals[ind[1]])+" is given by t0 = "+str(t0_vals[ind[0]])+" and t1 = "+str(t1_vals[ind[1]]))
ind= find_min_indexes(mse_ridge)
print(r"Ridge: Best learning rate $\eta_0$ = "+str(t0_vals[ind[0]]/t1_vals[ind[1]])+" is given by t0 = "+str(t0_vals[ind[0]])+" and t1 = "+str(t1_vals[ind[1]]))
    
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(mse_ols, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(mse_ridge, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
