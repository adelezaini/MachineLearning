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

class NeuralNetwork:
      """ ....FFNN backprop... """
      def __init__(self, n_hidden_layers, n_hidden_neurons, hidden_activation, output_activation):
          " Create the architecture of the network "
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
