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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def FrankeFunction(x,y):
    """Evaluate the Franke Function: a two-variables function to create the dataset of vanilla problems"""
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
 
def Plot_FrankeFunction(x,y,z, title="Dataset"):
    """3D plot, suitable for plotting the Franke Function"""
    
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection='3d')
    #ax = fig.gca(projection="3d")

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
    
    
def create_xyz_dataset(n, mu, sigma):
    """ Create xyz dataset from the FrankeFunction with a added normal distributed noise.
    x,y variables are taken evenly distributed in the interval [0,1]
    
    Args:
    n (int): squared root of total number of datapoints
    mu (float): mean value of the normal distribution of the noise
    sigma (float): standard deviation of the normal distribution of the noise

    Returns x,y,z values, mashed on a grid.
    """
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)

    x,y = np.meshgrid(x,y)
    z = FrankeFunction(x,y) + mu + sigma * np.random.randn(n,n)
    
    return x,y,z

def create_X(x, y, n = None):
    """Design matrix for two indipendent variables x,y.
        If n = None, return X as expected from the Neural Network."""
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    if n != None:
        N = len(x)
        l = int((n+1)*(n+2)/2)		# Number of elements in beta, number of feutures (degree of polynomial)
        X = np.ones((N,l))

        for i in range(1,n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)
    else:
        X = np.transpose(np.concatenate([x[np.newaxis,:],y[np.newaxis,:]],axis = 0))

    return X

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
