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

from numpy.core.function_base import logspace
from regan import *
import numpy as np

savefigures = True

np.random.seed(1234)

# Datapoints (squared root of datapoints -> meshgrid)
n = 30
# Paramaters of noise distribution
mu_N = 0; sigma_N = 0.2
# Max degree of complexity
maxdegree = 20

# Create vanilla dataset:
x,y,z = create_xyz_dataset(n,mu_N, sigma_N)

lambdas = [10**x for x in [-12, -6, -3, 0, 3]]

run_plot_compare(z, 100, N=n, k=5,poly_degree = 18,plot=True, saveplots=savefigures, foldername = 'Task5')

compare_lmd_BS(z, n, lambdas, maxdegree, solver = 'RIDGE', n_resampling = 100, saveplots = savefigures, foldername = 'Task4')
compare_lmd_CV(z, n, 5, lambdas, maxdegree, solver = 'RIDGE', saveplots = savefigures, foldername = 'Task4')
compare_lmd_CV(z, n, 10, lambdas, maxdegree, solver = 'RIDGE', saveplots = savefigures, foldername = 'Task4')

compare_lmd_BS(z, n, lambdas, maxdegree, solver = 'LASSO', n_resampling = 100, saveplots = savefigures, foldername = 'Task5')
compare_lmd_CV(z, n, 5, lambdas, maxdegree, solver = 'LASSO', saveplots = savefigures, foldername = 'Task5')
compare_lmd_CV(z, n, 10, lambdas, maxdegree, solver = 'LASSO', saveplots = savefigures, foldername = 'Task5')

