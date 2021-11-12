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
from algorithm import gradient_of

#activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softmax']
ACTIVATIONS = {'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu, 'leaky_relu': leaky_relu, 'elu': elu, 'softmax': softmax, None: None}
OPTIMIZERS = ['SGD', 'ADAGRAD', 'RMS', 'ADAM', 'SGDM']
ETAS = ['static', 'schedule', 'invscaling', 'hessian']
DELTA = 1e-8

    
# cost function
# look at here : for cross entropy https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
# derivative of cost function
# predict -> logistic or linear . look at :https://github.com/UdiBhaskar/Deep-Learning/blob/master/DNN%20in%20python%20from%20scratch.ipynb

class NeuralNetwork:
    """This class creates a Feed Forward Neural Network with the backpropagation algorithm.

    Class members after initializing and fitting the class:
      - n_inputs (int): number of inputs (i.e. number of rows of X and Y)
      - n_features (int): number of feutures/examples (i.e. number of columns of X)
      - n_categories (int): number of categories (i.e. number of columns of Y)
      - hidden_layers_dims (int list): list of number of neurons for each hidden (!) layer (e.g. [2,3,4,2])
      - n_hidden_layers (int): number of hidden layer
      - layers_dims (int list): list of number of neurons for each layer (e.g. [n_feutures, 2,3,4,2, n_categories])
      - activations_list (string list): list of activation functions name of the hidden layers + output activation function (default: None). The output activation function chagnes in changing the derivative class to NN_Classifier ('softmax')
      - parameters (dict): optimized weights and biases (after trained the NN)
      - grads (dict): gradients (after back propagation)
      - costs (list): list of cost results step by step in the optimization process
      - alpha (float): parameter for 'elu' and 'leaky_relu' activation function
      - opt (string): optimizer ['SGD', 'ADAGRAD', 'RMS', 'ADAM', 'SGDM']
      - batch_size (int): batch size for the SGD
      - n_epochs (int): number of epochs for the SGD
      - b1, b2: parameters to set up the optimization algorithm
      - eta0 (float): basic parameter for many learning rate schedule
      - eta_type (string): name of the choosen learning rate schedule
      - t1 (float): paramater needed for the learning rate 'schedule'
      - penality, lmd: parameters to set the regularization routine
    """

    
########## INITIALIZE NEURAL NETWORK
    def __init__(self, hidden_layes_dims, hidden_activations = ['sigmoid'],
        alpha = 1e-2, batch_size = 64, n_epochs = 50, opt = 'SGD', b1 = 0.9, b2 = 0.999,
        eta0 = 0.1, eta_type = 'static', t1 = 50,
        penality = None, lmd = 0):
        """Initialize the NN means:
          - create the architecture (layers, neurons, activations functions) of the network,
          - set all the paramaters needed for the backpropagation and the optimization algorithms.
          - NOT initialize inputs X and output Y.
        
          Args:
          - hidden_layers_dims (list): list of number of neurons for each hidden (!) layer (e.g. [2,3,4,2])
          - hidden_activations (list): activation functions of the hidden layers. If len() = 1 and number of hidden layers > 1, it automatically sets a list of the same activation functions (* n_hidden_layers).
          - alpha (float): parameter for 'elu' and 'leaky_relu' activation function
          - opt, batch_size, n_epochs, b1, b2: parameters to set up the optimization algorithm
          - eta0, eta_type, t1: parameters to set up the learning rate
          - penality, lmd: parameters to set the regularization routine
        """
        
        ### Layers and neurons
        self.n_hidden_layers = len(hidden_layers_dims)
        self.hidden_layers_dims = hidden_layers_dims # list of n. of neurons in each hidden layer (no input nor output)

        #### Hidden activation functions:

          # Check if activations functions list has same lenght ad n_hidden_layers
        assert len(hidden_activations) == len(self.n_hidden_layers), "Lenght of 'hidden_activations' list doesn't match with 'hidden_layer' lenght."

          # Check if activation functions are within the list
        if all(hidden_activations[i] not in list(activations.keys()) for i in range(len(hidden_activations))):
            raise ValueError("Activation functions must be defined within "+str(list(activations.keys())))
        else:
            self.activations_list = hidden_activations
            
        #### Output activation function: (default: Regression)
        self.activations_list.append(None)

        ### Parameters
        self.alpha = alpha
        self.batch_size = batch_size; self.n_epochs = n_epochs
        self.opt = opt # Optimization algorithm ['SGD', 'ADAGRAD', 'RMS', 'ADAM', 'SGDM']

        if opt not in OPTIMIZERS:
            raise ValueError("Optimizer must be defined in "+str(OPTIMIZERS))
          
        self.b1 = b1; self.b2 = b2
        self.eta0 = eta0; self.eta_type = eta_type; self.t1 = t1

        if eta_type not in ETAS:
            raise ValueError("Learning rate type must be defined within "+str(ETAS))
        self.penality = penality; self.lmd = lmd
        
########## FIT THE NETWORK
    def fit(self, X, Y, seed = None):
        """Fit the network with the given input X and output Y.
        Initialiaze layers, parameters and train the network."""

        assert X.shape[0] == Y.shape[0], "Size of input is different of size of the output"

        self.n_categories = Y.shape[1]
        self.n_features = X.shape[1]
        self.n_inputs = X.shape[0] # = Y.shape[0]

        self.layers_dims = [self.n_feutures] + self.hidden_layers_dims + [self.n_categories] # [ n. of input features, n. of neurons in hidden layer-1,.., n. of neurons in hidden layer-n shape, output]

        # 0) INITIALIZE PARAMATERS (WEIGHTS AND BIASES)
        self.parameters = self.init_parameters(seed)

        # 1) TRAIN THE NETWORK
        return self.train(X, Y)
          
          
########## 0) INITIALIZE PARAMATERS (WEIGHTS AND BIASES)
    def init_parameters(self, seed):
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
        
        return parameters

########## 1) TRAIN THE NETWORK
    def train(self, X, Y): # i.e. SDG
        """ The core algorithm of the NN object:
        1) Feed Forward to arrive to the output AL
        2) Backpropagation to evaluate all the grads for each example/feuture
        3) Update parameters (weights and biases) according to the optimizer (throughout the examples/feutures)
        4) All wrapped up in the STOCASTIC part (i.e. epochs and mini batches) of the SGD algorithm
        """
        
        self.grads = {}
        self.costs = []
        
        idx = np.arange(0, self.n_features)
        count=0
        
        for epoch in range(1,self.n_epochs+1):
            np.random.shuffle(idx)
            X = X[:,idx]
            Y = Y[:,idx]
            for i in range(0, self.n_features, self.batch_size):
                count += 1
                X_batch = X[:,i:i + self.batch_size]
                Y_batch = Y[:,i:i + self.batch_size]
  
                # a) Forward propagation
                AL, cache = self.forward_propagation(X_batch)
                
                # b) Compute cost function
                cost = self.compute_cost(AL, Y_batch)
                self.costs.append(cost)
                
                # c) Back propagation
                grads = self.back_propagation(AL, Y_batch, cache)
                self.grads = grads
                            
                # d) Update parameters (weights and biases)
                eta = self.learning_rate(X, self.eta0, self.eta_type, t = (epoch * self.batch_size + i), t1 = self.t1)
                self.parameters = self.update_opt_parameters(grads, eta)

        return self
        
########## 1a) Forward propagation:
    def feed_forward(self, X):
        """"
        Feed Forward algorithm
        
        Args:
            - X (array/matrix): input data of shape (input size, number of examples/feutures)
            
        Returns:
            - AL (array/matrix): output layer
            - caches (list): 'linear_activation_forward' (A_(l-1), Wl, bl, Zl), dim = [0,L-1]
        """
        Al = X
        L = self.n_hidden_layers
        caches = []
        
        for l in range(1, L): # [1,...,L-1]
        # NB: W1 = X (A0) --(1)--> A1
        #     WL = A(L-1) --(L-1)--> O(AL)
        #     len(activations_list) = L-1 ([0,L-2]
        
            A_prev = Al
            Wl = self.parameters['W' + str(l)]
            bl = self.parameters['b' + str(l)]
            Zl = np.dot(Wl, A_prev) + bl
            
            #Evaluate the next layer
            act_func = ACTIVATIONS[self.activations_list[l-1]] #hidden_activation(Z, l)
            Al = act_func(Zl, self.alpha).eval()
            
            # Record all the elements of evaluating each layer (A_(l-1), Wl + bl = Zl, A_(l-1) + Zl = Al ***yuppy***)
            cache = (A_prev, Wl, bl, Zl)
            caches.append(cache)

        AL = Al # no need but it helps to read it
        return AL, caches

    def predict(self, X):
        raise NotImplementedError("Method NeuralNetwork.predict is abstract and cannot be called.")

########## 1b) Compute cost function:
    def cost_function(self, A, Y):
        raise NotImplementedError("Method NeuralNetwork.cost_function is abstract and cannot be called.")
              
    def compute_cost(self, AL, Y):
        """
        Implement the cost function with optional regulation tecniques 'l1' and 'l2'
        
        Args:
          - AL (array/matrix): post-activation, probability vector, output of forward propagation
          - Y (array/matrix): "true" labels vector, of shape (output size, number of examples)
        
        Returns:
          - cost (float): value of the regularized loss function
        """

        cost = cost_function(AL,Y)
        
        L = len(self.parameters)//2
        
        if penality == 'l2' and lmb != 0:
            sum_weights = 0
            for l in range(1, L):
                sum_weights = sum_weights + np.sum(np.square(self.parameters['W' + str(l)]))
            cost = cost + sum_weights * (lmb/(2*m))
        elif penality == 'l1' and lmb != 0:
            sum_weights = 0
            for l in range(1, L):
                sum_weights = sum_weights + np.sum(np.abs(self.parameters['W' + str(l)]))
            cost = cost + sum_weights * (lmb/(2*m))
    
        return cost

########## 1c) Back propagation:
    def back_propagation(self, AL, Y, cache):
        """Implement the backward propagation
    
        Args:
            AL (array/matrix): probability vector, output of the forward propagation
            Y (array/matrix): true "label" vector (containing 0 if not the right digit, 1 if right digit)
            
        Returns:
             grads (dict): gradients of A, W, b for each HIDDEN layer (no cost function yet)
               grads["dAl"] = ...
               grads["dWl"] = ...
               grads["dbl"] = ...
        """
        
        grads = {}
        L = len(self.layer_dims) # the number of layers
    
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
    
        # Error of the output layer (initializes the backprop alg)
        dZL = gradient_of(cost_function, AL, Y)
        """#dZL = AL - Y # here derivative pf cost function!!!
        #delO = self.sigmoid_derivative(self.output)*(self.output - targets)*sum(inputs.T)"""
    
        AL, W, b, Z = caches[L-1]
        grads["dW" + str(L)] = np.dot(dZL,AL.T)/m
        grads["db" + str(L)] = np.sum(dZL,axis=1,keepdims=True)/m
        grads["dA" + str(L-1)] = np.dot(W.T,dZL)
    
        for l in reversed(range(L-1)): # Loop from l=L-2 to l=0
        
            A_prev, Wl, bl, Zl = caches[l]
            
            m = A_prev.shape[1]
            dA_prev = grads["dA" + str(l + 1)]
            
            act_func = ACTIVATIONS[self.activations_list[l-1]] # not sureeeeee of l-1
            dAl = act_func(Zl,self.alpha).grad()
            dZl = np.multiply(dA_prev,dAl)
            grads["dA" + str(l)] = np.dot(W.T,dZl)
            grads["dW" + str(l + 1)] = np.dot(dZl,A_prev.T)/m
            grads["db" + str(l + 1)] = np.sum(dZl,axis=1,keepdims=True)/m
            
            if penality == 'l2':
                grads["dW" + str(l + 1)] += ((lamda * Wl)/m)
            elif penality == 'l1':
                grads["dW" + str(l + 1)] += ((lamda * np.sign(Wl+DELTA))/m)

        return grads
        
########## 1d) Choose the learning rate schedule
    def learning_rate(self, X, eta0, eta_type, t, t1):
        """
        Update the learning rate when it is different than a given costant value
        Test that the algorithm converges (comparison with Hessian matrix eigenvalues)
        
        Args:
          - X (array/matrix): input data, needed to evaluate the Hessian matrix
          - eta0 (float): basic parameter for many learning rate schedule
          - eta_type (string): name of the choosen learning rate schedule
          - t (float): uptdating parameters to evaluate dynamic learning rate
          - t1 (float): paramater needed for the learning rate 'schedule'
          
        Returns:
          - eta (float): learning rate
        """
    
        # Evaluate the hessian metrix to test eta < max H's eigvalue
        H = (2.0/X.shape[0])* (X.T @ X)
        eigvalues, eigvects = np.linalg.eig(H)
        eta_opt = 1.0/np.max(eigvalues)
        eta = eta_opt
        
        if eta_type == 'static':
            eta = eta0
        elif eta_type == 'schedule':
            t0 = eta0 * t1
            eta = learning_schedule(t, t0=t0, t1=t1)
        elif eta_type == 'invscaling':
            power_t = 0.25 # one can change it but I dont want to overcrowd the arguments
            eta = eta0 / pow(t, power_t)
        elif eta_type == 'hessian':
            pass
            
        assert eta > eta_opt, "Learning rate higher than the inverse of the max eigenvalue of the Hessian matrix: SGD will not converge to the minimum. Need to set another learning rate or its paramentes."
        
        return eta
        
        
########## 1d) Update parameters (weights and biases)
    def update_opt_parameters(self, grads,eta):
        """
        Update parameters using gradient descent algorithm with different optimizers.
        OBS: The "Stochastic" is in the external loop
    
        Args:
        - grads (dict): output gradients from BP
    
        Returns:
        - parameters (dict): updated parameters
        """
        
        # To make it easier to go through the code:
        parameters = self.parameters
        b1 = self.b1; b2 = self.b2
        
        L = len(parameters) // 2 # number of layers in the neural network
                
        # Initialize optimization paramaters:
        for l in range(1, len(self.layer_dims)):
            opt_parameters['vdw' + str(l)] = np.zeros((self.layer_dims[l], self.layer_dims[l-1]))
            opt_parameters['vdb' + str(l)] = np.zeros((self.layer_dims[l], 1))
            opt_parameters['sdw' + str(l)] = np.zeros((self.layer_dims[l], self.layer_dims[l-1]))
            opt_parameters['sdb' + str(l)] = np.zeros((self.layer_dims[l], 1))
            
        for l in range(L):
            # Just rewrite to make the code easier to follow
            W = parameters["W" + str(l+1)]
            dW = grads["dW" + str(l + 1)]
            b = parameters["b" + str(l+1)]
            db = grads["db" + str(l + 1)]
            
            if method == 'SGD':
                W -= eta * dW
                b -= eta * db
                
            else:
                vdw = opt_parameters['vdw'+str(l+1)]
                vdb = opt_parameters['vdb'+str(l+1)]
                sdw = opt_parameters['sdw'+str(l+1)]
                sdb = opt_parameters['sdb'+str(l+1)]
                
                if method == 'SGDM':
                
                    vdb = b1 * vdb - eta * db
                    vdw = b1 * vdw - eta * dW
                    
                    W += vdw
                    b += vdb
                    
                elif method == 'ADAGRAD':
                
                    sdb = sbd + np.square(db)
                    sdw = sdw + np.square(dW)
                    
                    W -= eta * (dW/(np.sqrt(sdw) + DELTA))
                    b -= eta *(db/(np.sqrt(sdb) + DELTA))
                    
                elif method == 'RMS':
                
                    sdb = b1 * sbd + (1. - b1) * np.square(db)
                    sdw = b1 * sdw + (1. - b1) * np.square(dW)
                    
                    W -= eta * (dW/(np.sqrt(sdw) + DELTA))
                    b -= eta *(db/(np.sqrt(sdb) + DELTA))
                    
                elif method == 'ADAM':
                    vdb = (b1 * vdb + (1.-b1) * db)/(1.-b1)
                    vdw = (b1 * vdw + (1.-b1) * dW)/(1.-b1)
                    sdb = (b2 * sdb + (1.-b2) * np.square(db))/(1.-b2)
                    sdw = (b2 * sdw + (1.-b2) * np.square(dW))/(1.-b2)
                    
                    W -= eta * (vdw/(np.sqrt(sdw) + DELTA))
                    b -= eta * (vdb/(np.sqrt(sdb) + DELTA))
                    
          # Update back the parameters in the dictionaries
                opt_parameters['vdw'+str(l+1)] = vdw
                opt_parameters['vdb'+str(l+1)] = vdb
                opt_parameters['sdw'+str(l+1)] = sdw
                opt_parameters['sdb'+str(l+1)] = sdb
                                                          
            parameters["W" + str(l+1)] = Wl
            grads["dW" + str(l + 1)] = dWl
            parameters["b" + str(l+1)] = bl
            grads["db" + str(l + 1)] = dbl
            
        self.parameters = parameters

        return parameters
        

##################### NN_Regression #############################################
class NN_Regression(NeuralNetwork):
    """Derivative class of NeuralNetwork that implements methods for Linear Regression:
        - output activation function = None (default)
        - cost function = MSE
        - predict = returns function prediction
    """
    def cost_function(A, Y): #MSE
          return mean_squared_error(A,Y)
          
    def predict(self, X): #check!!!
        ''' Predicting values with FF (with no parameters optimization)

        Args: X (array/matrix): input data of shape (input size, number of examples/feutures)
        Returns: Y (array/matrix): output values
        '''
        
        out, _ = self.forward_propagation(X)
        return out.T
          
##################### NN_Classifier #############################################
class NN_Classifier(NeuralNetwork):
    """Derivative class of NeuralNetwork that implements methods for Logistic Regression:
        - output activation function = 'softmax'
        - cost function = Cross-Entropy
        - predict = can return the probability of occurance or the index (i.e. digit) of the maximum probability in n_categories (Y columns).
    """
    def __init__(self):
        super().__init__(X)
        self.activations_list[-1] = 'softmax'
        
    def cost_function(A, Y): # Cross_Entropy
          return np.squeeze(-np.sum(np.multiply(np.log(A),Y))/Y.shape[1])
          
    def predict(self, X, proba=False): #check!!!
        ''' Predicting values (with no parameters optimization)

        Args:
            - X (array/matrix): input data of shape (input size, number of examples/feutures)
            - prob (bool): True - return function values
                           False – return filtered best likelihood
            
        Returns:
            - Y (array/matrix): output values, as probability values or index (i.e. digit) of n_categories (Y columns).
        '''
        
        out, _ = self.forward_propagation(X,self.hidden_layers,self.parameters,self.keep_proba,self.seed)
        if proba == True:
            return out.T
        else:
            # Obtain prediction by taking the class with the highest likelihood
            return np.argmax(out, axis=0) #check!!!


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
"""
      #hidden_activation = 'sigmoid', output_activation=None, activations = None,"""
      
              

