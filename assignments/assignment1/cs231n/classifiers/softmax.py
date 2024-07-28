from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_features = X.shape[1]
    num_classes = W.shape[1]
    scores = np.zeros((num_train, num_classes))
    probabilities = np.zeros_like(scores)
    loss_sample = np.zeros(num_train)
    
    for i in range(num_train):
        scores[i, :] = X[i, :].dot(W)
        # Shift values so that highest class probability score is zero to improve numerical stability.
        scores[i, :] -= np.max(scores[i, :])
        
        # Calculate exponentiated probability row sum for normalisation calculation
        scores_row_sum = np.sum(np.exp(scores[i, :]))
        
        for j in range(num_classes):
            # Convert probability scores to normalised probabilities
            scores[i, j] = np.exp(scores[i,j])/scores_row_sum
            
            # From derivations, we know that dL/dW = (normalised probability - 1) * X (for j = y_i)
            # and dL/dW = normalised probability * X (for j != y_i) 
            
            # Get normalised probabilities from scores array, subtracting 1 if the class is equal to the correct class
            probabilities[i, j] = scores[i, j]
            if j == y[i]:
                probabilities[i, j] -= 1
   
        
        # Multiply X (input) with normalised probability to get dL/dW (dW)
        # X.T (F x 1) * Prob (1 x C) = (F x C)
        dW += np.dot(X[i, :].T.reshape(num_features,1), probabilities[i, :].reshape(1, num_classes))
        
        # Sample loss is calculated using the log of the normalised probability of the correct class
        loss_sample[i] = -np.log(scores[i, y[i]])
        
    # To get full loss, compute average loss per sample and add regularisation loss.
    loss = np.sum(loss_sample) / num_train
    loss += reg * np.sum(W * W)

    # Do the same for the gradient.
    dW /= num_train
    dW += 2 * reg * W
              

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = np.zeros((num_train, num_classes))
    
    scores = X.dot(W)

    # Shift values so that highest class probability score is zero to improve numerical stability.
    scores -= np.max(scores, axis=1).reshape(num_train, 1)

    # Convert probability scores to normalised probabilities
    scores = np.exp(scores)/np.sum(np.exp(scores), axis=1).reshape(num_train, 1)

    # Sample loss is calculated using the log of the normalised probability of the correct class
    loss_sample = -np.log(scores[np.arange(num_train), y])
    
    # To get full loss, compute average loss per sample and add regularisation loss.
    loss = np.sum(loss_sample) / num_train
    loss += reg * np.sum(W * W)
    
    # From derivations, we know that dL/dW = dL/df * df/dW = dL/df * X
    # dscores (df) = dL/df = (normalised probability - 1) (for j = y_i)
    # dL/df = normalised probability (for j != y_i)
    # Here, we get the normalised probabilities calculated previously and subtract 1 from the correct class scores
    dscores = scores
    dscores[np.arange(num_train), y] -= 1
    
    # Multiply X (input) with normalised probability to get dL/dW (dW)
    # X.T (F x 1) * Prob (1 x C) = (F x C)
    dW += np.dot(X.T, dscores)
    
    # Do the same for the gradient.
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
