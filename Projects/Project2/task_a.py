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
from dataset import create_xyz_dataset, create_X
from regression import


np.random.seed(1234)

# Degree of the polynomial
degree=5
# Datapoints (squared root of datapoints -> meshgrid)
n = 25
# Paramaters of noise distribution
mu_n = 0; sigma_n = 0.1
# Parameter of splitting data
test_size = 0.2

# Create vanilla dataset:
x,y,z = create_xyz_dataset(n,mu_n, sigma_n)
Plot_FrankeFunction(x,y,z, title="Original dataset: : \nFranke Function with stochastic noise")
z = z.ravel()
X = create_X(x, y, degree)

model=OLSRegression(X,z)
model.split()
model.rescale()
beta = model.fit()
z_tilde = model.predict_train()
z_predict = model.predict_test()


