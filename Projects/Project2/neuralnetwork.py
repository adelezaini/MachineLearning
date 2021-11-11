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
from activation import sigmoid, tanh, relu, leaky_relu, elu, softmax
from sklearn.metrics import mean_squared_error

#activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softmax']
ACTIVATIONS = {'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu, 'leaky_relu': leaky_relu, 'elu': elu, 'softmax': softmax, None: None}
    
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
      def __init__(self, regression = 'Linear', hidden_layes_dims, hidden_activations = ['sigmoid'],
      #hidden_activation = 'sigmoid', output_activation=None, activations = None,
          alpha = 1e-2):
                  #lmd = 0.,  seed = 1234,
                  #opt = 'SGD', M = 64, n_epochs = 50):
          """Create the architecture of the network:
          
            Args:
            - regression (string): 'Linear' or 'Logistic'. Set output activation function to 'None' or to 'softmax' rispectively.
            - hidden_layers_dims (list): list of number of neurons for each hidden layer (e.g. [2,3,4,2])
            - hidden_activations (list): activation functions of the hidden layers. If len() = 1 and number of hidden layers > 1, it automatically sets a list of the same activation functions (* n_hidden_layers).
            - alpha (float): parameter for 'elu' and 'leaky_relu' activation function
            - opt, M, n_epochs: parameters to set up the algorithms
          """
          
          self.n_hidden_layers = len(hidden_layers_dims)
          self.hidden_layers_dims = hidden_layers_dims
          
          # Hidden activation functions:
          assert(len(hidden_activations) == len(self.n_hidden_layers), "Lenght of 'hidden_activations' list doesn't match with 'hidden_layer' lenght."
          if all(hidden_activations[i] not in list(activations.keys()) for i in range(len(hidden_activations))):
              raise ValueError("Activation functions must be defined within "+str(list(activations.keys())))
          else:
              self.activations_list = hidden_activations
              
          #Output activation function: (default: Regression)
          self.activations_list.append(None)
          
          
    def init_weights_bias(seed):
        """ Initialize the weights and the biases in the given achitecture (i.e. [2,4,3,2] as list of number of neurons), using a normal distribution N(0,sqrt(2/fanin))."
        Args:
            seed (int): random seed to generate weights
        Returns:
            parameters (dict): dictionary containing weights matrixes and biases vectors, named as "W1", "b1", ..., "WL", "bL":
                    - Wl: weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    - bl: bias vector of shape (layer_dims[l], 1)
        """
        if seed != None:
            np.random.seed(seed)
            
        parameters = {}
        L = len(self.layer_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.normal(0,np.sqrt(2.0/self.layer_dims[l-1]),(self.layer_dims[l], self.layer_dims[l-1]))
            parameters['b' + str(l)] = np.random.normal(0,np.sqrt(2.0/self.layer_dims[l-1]),(self.layer_dims[l], 1))
          self.parameters = parameters
          
              
    """def fit(self, X, Y, seed = None):
    
      assert(X.shape[0] == Y.shape[0], "Size of input is different of size of the output")
      
      self.n_categories = Y.shape[1]
      self.n_features = X.shape[1]
      self.n_inputs = X.shape[0]
      
      self.X = X
      self.Y = Y
      
      self.layers_dims = [self.n_feutures, self.hidden_layers_dims, self.n_categories] # [ n. of input features, n. of neurons in hidden layer-1,.., n. of neurons in hidden layer-n shape, output]
      
      self.init_weights_bias(seed)"""
          
    def feed_forward(self,X):
        
        A = X
        L = len(self.layer_dims)
        for l in range(1, L): # [1,...,L-1]
        # NB: W1 = X (H0) --(1)--> H1
        #     WL = H(L-1) --(L-1)--> O(HL)
        #     len(activations_list) = L-1 ([0,L-2]
        
            Wl = parameters['W' + str(l)]
            bl = parameters['b' + str(l)]

            Z = np.dot(Wl, A) + bl
            act_func = ACTIVATIONS[self.activations_list[l-1]] #hidden_activation(Z, l)
            A = act_func(Z)
        
        return A
              
    def compute_cost(A, Y, parameters, penality=None, lmb=0):
        """
        Implement the cost function with None,
        
        Arguments:
        A -- post-activation, output of forward propagation
        Y -- "true" labels vector, of shape (output size, number of examples)
        parameters -- python dictionary containing parameters of the model
        
        Returns:
        cost - value of the regularized loss function
        """

        cost = cost_function(A,Y)
        
        L = len(parameters)//2
        
        if penality == 'l2' and lmb != 0:
            sum_weights = 0
            for l in range(1, L):
                sum_weights = sum_weights + np.sum(np.square(parameters['W' + str(l)]))
            cost = cost + sum_weights * (lmb/(2*m))
        elif penality == 'l1' and lmb != 0:
            sum_weights = 0
            for l in range(1, L):
                sum_weights = sum_weights + np.sum(np.abs(parameters['W' + str(l)]))
            cost = cost + sum_weights * (lmb/(2*m))
    
    return cost
    
    def predict(self,X):
        pass

class NN_Regression(NeuralNetwork):

      def cost_function(A, Y): #MSE
          return mean_squared_error(A,Y)
        
class NN_Classifier(NeuralNetwork):
    def __init__(self):
        super().__init__(X)
        self.activations_list[-1] = 'softmax'
        
      def cost_function(A, Y): # Cross_Entropy
          return np.squeeze(-np.sum(np.multiply(np.log(A),Y))/Y.shape[1])


    """
    def hidden_activation(Z, l):
        Condense if-conditions of the activation functions.
        i =l - 1
        if self.activations[i] == 'sigmoid':
            return sigmoid(Z)
        elif self.activations[i] == 'tanh':
            return tanh(Z)
        elif self.activations[i] == 'relu':
            return relu(Z)
        elif self.activations[i] == 'leaky_relu':
            return leaky_relu(Z)
        elif self.activations[i] == 'elu':
            return elu(Z)
        elif self.activations[i] == 'softmax':
            return softmax(Z)
            
  activations = {'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu, 'leaky_relu': leaky_relu, 'elu': elu, 'softmax': softmax}
        """
    """
                1) Same hidden activation function approach:
            - hidden_activation (string): activation function of the hidden layers (default: sigmoid)
            - output_activation (string): activation function of the output layer. If None, it means that we are dealing with Linear Regression, otherwise it is Logistic Regression (recommended 'softmax').
            
            2) Different hidden activation function approach:
            - activations (list): list of activations functions (hidden and output)
            
                      if activations = None:
              if hidden_activation not in list(activations.keys()):
                  raise ValueError("Hidden activation function must be defined within "+str(list(activations.keys())))
              #else:self.hidden_activation = hidden_activation
                  
              if output_activation not in list(activations.keys()):
                  raise ValueError("Output activation function must be defined within "+str(list(activations.keys())))
              #else:self.output_activation = output_activation
                  
              self.activations_list = [hidden_activation] * self.n_hidden_layers
              self.activations_list[-1] = output_activation
              
          else: # to test
              assert(len(activations) == (len(self.n_hidden_layers)+1), "Lenght of 'activations' list doesn't match with 'hidden_layer' lenght (+1)."
              if activations[:] not in list(activations.keys()):
                  raise ValueError("Activation functions must be defined within "+str(list(activations.keys())))
              self.activations_list = activations
              
              
                        # Output activation function:
          if regression not in ['Linear, Logistic']:
              raise ValueError("Regression must be defined as 'Linear' or 'Logistic'.")
          if regression == 'Linear':
              self.activations_list.append(None)
          elif regression == 'Logistic':
              self.activations_list.append('softmax')
            """

              

