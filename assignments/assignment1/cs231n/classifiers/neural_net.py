from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

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
        Compute the loss and gradients for a two layer fully connected neural
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
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # first layer output = X * W1 + b1
        first_layer_output = X.dot(W1) + b1

        # First layer activation = ReLU(first layer output)
        first_layer_activation = np.maximum(first_layer_output, 0)

        # Second layer output = first layer activation * W2 + b2
        second_layer_output = first_layer_activation.dot(W2) + b2
        scores = second_layer_output

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Shift values so that highest class probability score is zero to improve numerical stability.
        softmax_scores = scores
        softmax_scores -= np.max(softmax_scores, axis=1).reshape(N, 1)

        # Convert probability scores to normalised probabilities
        softmax_scores = np.exp(softmax_scores)/np.sum(np.exp(softmax_scores), axis=1).reshape(N, 1)

        # Sample loss is calculated using the log of the normalised probability of the correct class
        loss_sample = -np.log(softmax_scores[np.arange(N), y])

        # To get full loss, compute average loss per sample and add regularisation loss.
        loss = np.sum(loss_sample) / N
        loss += reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # f = X*W + b => df/dW = X
        # dL/dW = dL/df * df/dW = dL/df * X
        # From derivations, we know that dL/df = (normalised probability - 1) (for j = y_i)
        # dL/df = normalised probability (for j != y_i)
        # So, retrieve normalised probabilities calculated previously and subtract 1 from the correct class scores
        dscores = softmax_scores
        dscores[np.arange(N), y] -= 1

        # Backprop the gradient to the parameters, beginning with W2 and b2
        # dW2 = dL/dW2 = dL/df * df/dW2 = upstream gradient * local gradient
        # dL/df (upstream gradient) = dscores
        # f = A(L1)*W2 + b2 => df/dW2 (local gradient) = A(L1), A(L1) = first_layer_activation (fla)
        dW2 = np.dot(first_layer_activation.T, dscores)
        
        # db2 = dL/db2 = dL/df * df/db2
        # dL/df (upstream gradient) = dscores
        # f = A(L1)*W2 + b2 => df/db2 (local gradient) = 1
        # Sum axis=0 to get values for db2 across all classes
        db2 = np.sum(dscores, axis=0)
        
        # d_fla is our gradient wrt the first layer output, which we backprop as the new upstream gradient
        # d_fla = dL/d_fla = dL/df * df/d_fla
        # dL/df (upstream gradient) = dscores
        # f = A(L1)*W2 + b2 => df/d_fla (local gradient) = W2
        d_first_layer_activation = np.dot(dscores, W2.T)
        
        # NB: ReLU activation function has a gradient of 0 for input less than 0
        d_first_layer_activation[first_layer_activation <= 0] = 0
        
        # Backprop next layer into W1, b1
        # dW1 = dL/dW1 = dL/df * df/dW1
        # dL/df (upstream gradient) = d_fla
        # f = fla = max(X*W1 + b1, 0) => df/dW1 (local gradient) = X (f > 0) ; 0 (f <= 0) => matches d_fla
        dW1 = np.dot(X.T, d_first_layer_activation)
        
        # db1 = dL/db1 = dL/df * df/db1
        # dL/df (upstream gradient) = d_first_layer_activation
        # f = max(X*W1 + b1, 0) => df/db1 (local gradient) = 1 (f > 0) ; 0 (f <= 0) => matches d_fla
        # Sum axis=0 to get values for db1 across all classes
        db1 = np.sum(d_first_layer_activation, axis=0)
        
        # Normalise gradients by number of training samples and add regularisation
        dW1 /= N
        dW1 += 2 * reg * W1
        db1 /= N
        
        dW2 /= N
        dW2 += 2 * reg * W2
        db2 /= N
        
        # Store gradients
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

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
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
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
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            batch_int = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_int]
            y_batch = y[batch_int]

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
            
            # Retreive initialised values of parameters
            W1, b1 = self.params['W1'], self.params['b1']
            W2, b2 = self.params['W2'], self.params['b2']
            
            # Perform parameter update
            W1 += -learning_rate * grads['W1']
            b1 += -learning_rate * grads['b1']
            W2 += -learning_rate * grads['W2']
            b2 += -learning_rate * grads['b2']

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
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

        # first layer activation = ReLU(X*W1 + b1) = np.maximum(X * W1 + b1, 0)
        first_layer_activation = np.maximum(X.dot(self.params['W1']) + self.params['b1'], 0)

        # y_scores = first layer activation * W2 + b2
        y_scores = first_layer_activation.dot(self.params['W2']) + self.params['b2']
        
        # Get the index of each row's maximum score (i.e. the class)
        y_pred = np.argmax(y_scores, axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
