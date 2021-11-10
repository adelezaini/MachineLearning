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
from algorithm import SGD
from activation import sigmoid, tanh, relu, leaky_relu, elu

activations = ["sigmoid", "tanh", "relu", "leaky_relu", "elu"]

def init_weights_bias(...):
    Wi=np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) /np.sqrt(2.0/INPUT_LAYER_SIZE) #matrix of random numbers drawn from a normal distribution with mean 0 and variance 1.
    Wh=np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) /np.sqrt(2.0/HIDDEN_LAYER_SIZE)
    Wo=np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) /np.sqrt(2.0/HIDDEN_LAYER_SIZE)
    
    Bh = np.full((1, HIDDEN_LAYER_SIZE), 0.1)
    Bo = np.full((1, OUTPUT_LAYER_SIZE), 0.1)
    
# cost function
# look at here : for cross entropy https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
# derivative of cost function
# input of relu is Z (weighted input)
# set layers = list with number
# predict -> logistic or linear . look at :https://github.com/UdiBhaskar/Deep-Learning/blob/master/DNN%20in%20python%20from%20scratch.ipynb

class NeuralNetwork:
      """ ....FFNN backprop... """
      '''Parameters: layer_dims -- List Dimensions of layers including input and output layer
                hidden_layers -- List of hidden layers
                                 'relu','sigmoid','tanh','softplus','arctan','elu','identity','softmax'
                                 Note: 1. last layer must be softmax
                                       2. For relu and elu need to mention alpha value as below
                                        ['tanh',('relu',alpha1),('elu',alpha2),('relu',alpha3),'softmax']
                                        need to give a tuple for relu and elu if you want to mention alpha
                                        if not default alpha is 0
                init_type -- init_type -- he_normal  --> N(0,sqrt(2/fanin))
                             he_uniform --> Uniform(-sqrt(6/fanin),sqrt(6/fanin))
                             xavier_normal --> N(0,2/(fanin+fanout))
                             xavier_uniform --> Uniform(-sqrt(6/fanin+fanout),sqrt(6/fanin+fanout))
                                 
                learning_rate -- Learning rate
                optimization_method -- optimization method 'SGD','SGDM','RMSP','ADAM'
                batch_size -- Batch size to update weights
                max_epoch -- Max epoch number
                             Note : Max_iter  = max_epoch * (size of traing / batch size)
                tolarance -- if abs(previous cost  - current cost ) < tol training will be stopped
                             if None -- No check will be performed
                keep_proba -- probability for dropout
                              if 1 then there is no dropout
                penality -- regularization penality
                            values taken 'l1','l2',None(default)
                lamda -- l1 or l2 regularization value
                beta1 -- SGDM and adam optimization param
                beta2 -- RMSP and adam optimization value
                seed -- Random seed to generate randomness
                verbose -- takes 0  or 1
    '''
      def __init__(self, n_hidden_layers, n_hidden_neurons, hidden_activation, output_activation,
                  seed = 1234,
                  lmd = 0.,
                  opt = 'SGD', eta0 = 0.1, M = 64, n_epochs = 50): # Can add other
          "Create the architecture of the network"
          self.n_hidden_layers = n_hidden_layers
          self.n_hidden_neurons = n_hidden_neurons
          
          if hidden_activation not in activations:
              raise ValueError("Hidden activation function must be defined within "+str(activations))
          else:
              self.hidden_activation = hidden_activation
              
          if hidden_activation not in activations:
              raise ValueError("Output activation function must be defined within "+str(activations))
          else:
              self.output_activation = output_activation
              
    def fit(self, X, y):
          self.n_categories = y.shape[0]
          self.n_features = X.shape[1]
          
          self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
          self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01
          
          self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
          self.output_bias = np.zeros(self.n_categories) + 0.01
              

