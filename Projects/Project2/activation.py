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

class activation:
    """Abstract class to gathered all the activation functions."""
    
    def __init__(self, X, alpha = 1e-2):
        """Initialize the class to evaluate the actual result or the derivative's one
        
        Args:
          - X : input data
          - alpha: needed for 'elu' and 'leaky_relu'"""
          
        self.X = np.array(X,dtype=np.float64)
        self.alpha = alpha
        
    def eval(self):
        """Evaluate the result"""
        raise NotImplementedError("Method activation.eval is abstract and cannot be called")
        
    def gradient(self):
        """Evaluate the gradient"""
        raise NotImplementedError("Method activation.gradient is abstract and cannot be called")

class linear(activation):

    def eval(self):
        return self.X
        
    def grad(self):
        return np.ones(shape=self.X.shape)
        
        
class sigmoid(activation):

    def eval(self):
        return 1/(1 + np.exp(-self.X))
        
    def grad(self):
        return self.eval() * (1. - self.eval())
        
class tanh(activation):

    def eval(self):
        return np.tanh(self.X)
        
    def grad(self):
        return 1. - (np.tanh(self.X))**2
        
class relu(activation):
    def eval(self):
        return np.maximum(0., self.X)
        
    def grad(self):
        return np.where(self.X > 0, 1., 0.)

class leaky_relu(activation):

    def eval(self):
        return np.where(self.X > 0, self.X, self.alpha * self.X)
        
    def grad(self):
        return np.where(self.X > 0, 1., self.alpha)
        
class elu(activation):
        
    def eval(self):
        return np.where(self.X > 0, self.X, self.alpha * (np.exp(self.X)-1.))
        
    def grad(self):
        return np.where(self.X > 0, 1., self.alpha * np.exp(self.X))
        
class softmax(activation): # to test
        
    def eval(self):
        return np.exp(self.X) / np.sum(np.exp(self.X),axis=0)
      
    def grad(self):
        pass
      
    
  
"""
def elu_grad(x, alpha):
    return np.where(x > 0, 1., alpha * np.exp(x))
    
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def tanh(x):
    return np.tanh(x)
    
def relu(x):
    return np.maximum(0.,x)
    
def leaky_relu(x, alpha = 1e-2):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha):
    return np.where(x > 0, x, alpha * (np.exp(x)-1))
    
def sigmoid_grad(x):
    return sigmoid(x) * (1. - sigmoid(x))
    
def tanh_grad(x):
    return 1. - (np.tanh(x))**2
    
def relu_grad(x):
    return np.where(x > 0, 1., 0.)

def leaky_relu_grad(x, alpha = 1e-2):
    return np.where(x > 0, 1., alpha)
  
def elu_grad(x, alpha):
    return np.where(x > 0, 1., alpha * np.exp(x))
"""
