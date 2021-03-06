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

# Error analysis
def R2(y_data, y_model):
    """Compute the R2 score of the two given values"""
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    """Compute the Mean Square Error of the two given values"""
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def find_min_indexes(A):
    return np.array(np.where(A == A.min())).flatten()
    
def find_max_indexes(A):
    return np.array(np.where(A == A.max())).flatten()
    
def Rolling_Mean(vector, windows=3):
    """Evaluate the rolling mean of a vector
    
    Args:
    vector (array): input vector to evaluate the rolling mean
    window (int): integer indicating how many sequential datapoints to include in the rolling mean
    
    Returns:
    rolling mean of the given vector and two values at one sigma from the rolling average
    
    Usage:
        y_rm, y_down, y_up = Rolling_Mean(y,2)
        plt.plot(x, y_rm, color=color)
        plt.fill_between(x, y_down, y_up, alpha=0.1, color=color)
    """
    vector_df = pd.DataFrame({'vector': vector})
    # computing the rolling average
    rolling_mean = vector_df.vector.rolling(windows).mean().to_numpy()
    # computing the values at two sigmas from the rolling average
    rolling_std = vector_df.vector.rolling(windows).std().to_numpy()
    value_up = rolling_mean + rolling_std
    value_down = rolling_mean - rolling_std
    
    return rolling_mean, value_down, value_up
    


