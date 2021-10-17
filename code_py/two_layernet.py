from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """



    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two-layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2'] #shapes 10,3 -- 3
        N, D = X.shape

        # Compute the forward pass
        scores = 0.
        
        #############################################################################
        # TODO: Perform the forward pass, computing the class probabilities for the #
        # input. Store the result in the scores variable, which should be an array  #
        # of shape (N, C).                                                          #
        #############################################################################
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        z_2 = np.dot(X, W1) + b1
        a_2 = np.maximum(0, z_2)
        #print('Shape of a_2: ', a_2.shape)
        z_3 = np.dot(a_2, W2) + b2
        #print("dimentions of z_3",z_3.shape)

        def softmax(matrix):
            # find the maximum reshaping the array as vector 1d keeping the dimensionality as columns
            m = np.max(matrix, axis=-1, keepdims=True)
            #it can be show that accordingly to its formulation, the softmax function does not depens on the max
            sf_n = np.exp(matrix - m)
            sf_d = np.sum(sf_n, axis=-1, keepdims=True)
            return sf_n / sf_d

        scores = softmax(z_3)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        # If the targets are not given then jump out, we're done
        if y is None:
            return scores


        # Compute the loss
        loss = 0.
        
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        
        # Implement the loss for the softmax output layer
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        def L2_squared_norm(weights_matrix):
            # L2 squared norm can be derived directly using the equality ||W||^2 = <W1,W1>
            return weights_matrix.reshape(-1) * weights_matrix.reshape(-1)

        loss -= np.sum(np.log(np.choose(y, scores.T))) / np.shape(scores)[0]
        loss += reg * (np.sum(L2_squared_norm(W1)) + np.sum(L2_squared_norm(W2)))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}

        ##############################################################################
        # TODO: Implement the backward pass, computing the derivatives of the weights#
        # and biases. Store the results in the grads dictionary. For example,        #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size  #
        ##############################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Convert y vector of correct labels into a matrix of hot vectors
        y_lab_hot = np.zeros((y.size, y.max()+1))
        y_lab_hot[np.arange(y.size),y] = 1

        # Store results of loss gradient with respect to scores
        dJdz = (1/N)*(softmax(z_3) - y_lab_hot)

        # Intermediate chain rule gradients used to estimate final results
        dW1_1 = np.tensordot(dJdz, np.transpose(W2), axes=1)
        dW1_2 = dW1_1 * ((z_2 > 0) * 1)

        # Estimate final gradients and store in dictionary accordingly
        grads['W2'] = np.transpose(np.tensordot(np.transpose(dJdz), a_2, axes=1)) + 2 * reg * W2
        grads['b2'] = np.sum(dJdz, axis=0)
        grads['W1'] = np.tensordot(np.transpose(X), dW1_2, axes=1) + 2*reg*W1
        grads['b1'] = np.sum(dW1_2, axis=0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads



    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array of shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        
        num_train = X.shape[0]
        iterations_per_epoch = max( int(num_train // batch_size), 1)


        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = X
            y_batch = y

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            # Define the possible indexes to be shuffled (create sequence)
            indexes = np.arange(num_train)
            
            # Reorganize randomly the indexes
            np.random.shuffle(indexes)
            
            # Select the shuffled indexes for the minibatch to be used
            shuffled_indexes = indexes[0:batch_size-1]
            
            # Select the minibatch of the features
            X_batch = X[shuffled_indexes, :]
            
            # Select the minibatch  of the label
            y_batch = y[shuffled_indexes]
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
        
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            # Define update rule for W1
            self.params['W1'] += -learning_rate * grads['W1']
            
            # Define update rule for W2
            self.params['W2'] += -learning_rate * grads['W2']
            
            # Define update rule for b1
            self.params['b1'] += -learning_rate * grads['b1']
            
            # Define update rule for b2
            self.params['b2'] += -learning_rate * grads['b2']            

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # At every epoch check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }



    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Define the output for the hidden layer 1 ( z(2) )
        hidden_layer1 = np.dot(X, self.params['W1']) + self.params['b1']
        
        # Apply ReLu activation function ( phi(z(2)) )
        hidden_layer1[hidden_layer1<=0] = 0
        
        # Calculate scores based on first and second layer weights 
        scores = np.dot(hidden_layer1, self.params['W2']) + self.params['b2']
        
        # Make prediction taken the argmax of the scores
        y_pred = np.argmax(scores, axis=1)    

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred


