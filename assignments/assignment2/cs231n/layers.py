from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Reshape the input data into rows
    x_reshaped = np.reshape(x, (x.shape[0], -1))
    
    # output = X * W + b
    out = np.dot(x_reshaped, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Reshape the input data into rows
    x_reshaped = np.reshape(x, (x.shape[0], -1))
    
    # downstream gradient = upstream gradient (dout) * local gradient
    # f = X*W + b
    # df/dx = dx = W, df/dw = dw = X, df/db = 1
    dx = np.dot(dout, w.T).reshape(x.shape) # Need to reshape into same shape as x
    dw = np.dot(x_reshaped.T, dout)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # ReLU(x) = max(x, 0)
    out = np.maximum(x, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # ReLU activation function has a gradient of 0 for input less than 0
    dx = dout
    dx[x <= 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Compute sample mean and variance
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
    
        # Normalise input using the running mean and variance
        x_mean_centre = x - x_mean
        x_var_eps = np.sqrt(x_var + eps)
        x_norm = x_mean_centre/x_var_eps
        
        # Scale and shift the normalised data using gamma and beta
        out = x_norm * gamma + beta
        
        # Compute running averages for mean and variance by decaying via momentum
        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_var
        
        cache = x_mean_centre, x_var_eps, x_norm, gamma
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Normalise input using the running mean and variance
        x_norm = (x - running_mean)/np.sqrt(running_var + eps)
        
        # Scale and shift the normalised data using gamma and beta
        out = x_norm * gamma + beta
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Get values from cache
    x_mean_centre, x_var_eps, x_norm, gamma = cache
    
    N, D = x_norm.shape

    # From computational graph, we have:
    # X
    # a: mean(X) = 1/n * sum(X_i) from i=1 to N
    # b: X - mean(X) = X - a
    # c: (X - mean(X))**2 = b**2
    # d: var(X) = 1/n * sum(c_i) from i=1 to N
    # e: sqrt(var(X) + eps) = sqrt(d + eps)
    # f: 1/sqrt(var(X) + eps) = 1/e
    # g: x_norm = (X - mean(X))/sqrt(var(X) + eps) = b * f
    # h: x_norm * gamma = g * gamma
    # i: x_norm * gamma + beta = h + beta

    # dbeta (1 x D) = dL/di * di/dbeta = dout * 1
    dbeta = np.sum(dout, axis=0)
    # dh (N x D) = dL/di * di/dh = dout * 1
    dh = dout
    # dgamma (1 x D) = dL/dh * dh/dgamma = dh * g = dh * x_norm
    dgamma = np.sum(dh * x_norm, axis=0)
    # dg (N x D) = dL/dh * dh/dg = dh * gamma
    dg = dh * gamma
    # df (1 x D) = dL/dg * dg/df = dg * b
    df = np.sum(dg * x_mean_centre, axis=0)
    # de (1 x D) = dL/df * df/de = df * (-1/e**2)
    de = df * -1 / (x_var_eps)**2
    # dd (1 x D) = dL/de * de/dd = de * (1/sqrt(d + eps))
    dd = de * (1 / (2 * x_var_eps))
    # dc (N x D) = dL/dd * dd/dc = dd * ((1/n) * ones(N, D))
    dc = dd * ((1/N) * np.ones((N, D))) 
    # db (N x D) = dL/dc * dc/db + dL/dg * dg/db = dc * 2b + dg * f
    # = dc * 2(X - mean(X)) + dg * 1/sqrt(var(X) + eps)
    db = dc * 2 * x_mean_centre + dg * (1 / x_var_eps)
    # da (1 x D) = dL/db * db/da = db * (-1)
    da = np.sum(db * (-1), axis=0)
    # dx (N x D) = dL/da * da/dx + dL/db * db/dx = da * ((1/n) * ones(N, D)) + db * (1)
    dx = da * ((1/N) * np.ones((N, D))) + db * (1)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Initialise variables
    x_mean_centre, x_var_eps, x_norm, gamma = cache
    N, D = x_norm.shape
    
    # Compute gradients
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    dnorm = dout * gamma
    
    # From derivations:
    dx = ((dnorm * N) - x_norm * np.sum(dnorm * x_norm, axis=0) - np.sum(dnorm, axis=0))/(N * x_var_eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Initialise variables
    N,D = x.shape
    
    # Compute mean and variance layer-norm-wise (i.e. for each sample, axis=1)
    x_mean = np.mean(x, axis=1).reshape(N, -1)
    x_var = np.var(x, axis=1).reshape(N, -1)

    # Normalise input using the running mean and variance
    x_mean_centre = x - x_mean
    x_var_eps = np.sqrt(x_var + eps)
    x_norm = x_mean_centre/x_var_eps

    # Scale and shift the normalised data using gamma and beta
    out = x_norm * gamma + beta

    cache = x_mean_centre, x_var_eps, x_norm, gamma
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Initialise variables
    x_mean_centre, x_var_eps, x_norm, gamma = cache
    N, D = x_norm.shape
    
    # Compute gradients
    dgamma = np.sum(dout * x_norm, axis=0) # 1 x D, sum over N
    dbeta = np.sum(dout, axis=0) # 1 x D, sum over N
    dnorm = dout * gamma
    
    # From batchnorm_backward_alt derivations, and adjust:
    # axis=0 -> axis=1 to sum over D for each sample instead of N
    # Replacing N -> D to reflect the number of features D instead of N
    dx = ((dnorm * D) - x_norm * np.sum(dnorm * x_norm, axis=1).reshape(N, -1) - np.sum(dnorm, axis=1).reshape(N, -1))/(D * x_var_eps)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Apply mask to input to dropout neurons with probability p of survival
        # Divide by p so inference output doesn't need to be scaled by p when 
        # calculating expected value.
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # During backward pass, set the gradient weights using the corresponding
        # forward pass' mask to 0 as these nodes were dropped. 
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    P = conv_param['pad']
    S = conv_param['stride']
    
    # Zero pad input
    pad_width = ((0,0), (0,0), (P,P), (P,P))
    x_pad = np.pad(x, pad_width, 'constant')
    
    # Initialise output volume (i.e. stack of activation maps)
    # Output volume has shape (N, F, O, O), where O = (W-F+2P)/S + 1 
    OH = (H - HH + 2*P)//S + 1
    OW = (W - WW + 2*P)//S + 1
    out = np.zeros((N, F, OH, OW))
    
    # Compute multiplication of input area and filter (i.e. x*w + b)
    for n_i in range(N): # N data points
        for h_i in range(OH): # H height
            for w_i in range(OW): # W width
                for f_i in range(F): # F filters
                    out[n_i, f_i, h_i, w_i] = np.sum(x_pad[n_i, :, h_i*S:h_i*S+HH, w_i*S:w_i*S+WW] * w[f_i, :, :, :]) + b[f_i]
                    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    P = conv_param['pad']
    S = conv_param['stride']
    
    # Initialise gradients
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # Zero pad input and dx
    pad_width = ((0,0), (0,0), (P,P), (P,P))
    x_pad = np.pad(x, pad_width, 'constant')
    dx_pad = np.pad(dx, pad_width, 'constant')
    
    # Output volume has shape (N, F, O, O), where O = (W-F+2P)/S + 1 
    OH = (H - HH + 2*P)//S + 1
    OW = (W - WW + 2*P)//S + 1
    
    # Compute gradients
    for n_i in range(N): # N data points
        for f_i in range(F): # F filters
            db[f_i] += np.sum(dout[n_i, f_i, :, :])
            for h_i in range(OH): # H height
                for w_i in range(OW): # W width
                    dw[f_i, :, :, :] += x_pad[n_i, :, h_i*S:h_i*S+HH, w_i*S:w_i*S+WW] * dout[n_i, f_i, h_i, w_i]
                    dx_pad[n_i, :, h_i*S:h_i*S+HH, w_i*S:w_i*S+WW] += w[f_i, :, :, :] * dout[n_i, f_i, h_i, w_i]
                    
    # get dx from dx pad
    dx = dx_pad[:, :, P:P+H, P:P+W]
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Initialise variables
    N, C, H, W = x.shape
    HH, WW, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    
    # Initialise output with shape (N, C, H', W')
    PH = 1 + (H - HH)//S
    PW = 1 + (W - WW)//S
    out = np.zeros((N, C, PH, PW))
    
    # Compute max pooling
    for n_i in range(N): # N data points
        for c_i in range(C): # C channels
            for h_i in range(PH): # H height
                for w_i in range(PW): # W width
                    out[n_i, c_i, h_i, w_i] = np.max(x[n_i, c_i, h_i*S:h_i*S+S, w_i*S:w_i*S+S])
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Initialise variables
    x, pool_param = cache
    N, C, PH, PW = dout.shape
    S = pool_param['stride']
    dx = np.zeros_like(x)
    
    # For the backward pass for a max(x,y) operation, we route the gradient
    # only to the input that had the highest value in the forward pass.
    for n_i in range(N): # N data points
        for c_i in range(C): # C channels
            for h_i in range(PH): # H height
                for w_i in range(PW): # W width
                    x_curr = x[n_i, c_i, h_i*S:h_i*S+S, w_i*S:w_i*S+S] # current slice of input
                    i,j = np.unravel_index(np.argmax(x_curr), x_curr.shape) # location of maximum value in current slice
                    dx[n_i, c_i, h_i*S+i, w_i*S+j] = dout[n_i, c_i, h_i, w_i] # set values of dx to dout for the maximum values in the input
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Normal BN accepts inputs and produces output of shape (N, D).
    # For ConvLayers, we need it to accept input and produce output of shape (N, C, H, W).
    # Spatial BN computes mean and variance for each of the C feature channels by computing
    # statistics over the minibatch dimension N as well as the spatial dimensions H and W.
    
    # Initialise variables
    N, C, H, W = x.shape
    out = np.zeros_like(x)
    cache = {}
    
    # Compute batch normalisation statistics for each channel using flattened input
    for i in range(C):
        x_flat = x[:,i,:,:].flatten().reshape(-1,1)
        bn_out, cache[i] = batchnorm_forward(x_flat, gamma[i], beta[i], bn_param)
        out[:,i,:,:] = bn_out.reshape(N, H, W) # reshape flattened input

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Initialise variables
    N, C, H, W = dout.shape
    dgamma, dbeta = np.zeros(C), np.zeros(C)
    dx = np.zeros_like(dout)
    
    # Use flattened dout as input for backward pass, then reshape output.
    for i in range(C):
        dout_flattened = dout[:,i,:,:].flatten().reshape(-1,1)
        dx_flat, dgamma[i], dbeta[i] = batchnorm_backward(dout_flattened, cache[i])
        dx[:, i, :, :]= dx_flat.reshape(N, H, W) # reshape to original dimensions

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Initialise variables
    N, C, H, W = x.shape
    
    # Reshape X into (N',D'), allowing for G groups and flattening the H and W axes
    # We compute layer normalisation statistics for the N' = N*G samples and
    # D' = (C/G)*H*W dimensions
    g_shape = (N*G, (C//G)*H*W)
    x_reshape = x.reshape(g_shape)
    
    # Compute mean and variance layer-norm-wise (i.e. for each sample, axis=1)
    x_mean = np.mean(x_reshape, axis=1).reshape(N*G,-1)
    x_var = np.var(x_reshape, axis=1).reshape(N*G,-1)

    # Normalise input using the mean and variance
    x_mean_centre = x_reshape - x_mean
    x_var_eps = np.sqrt(x_var + eps)
    x_norm = x_mean_centre/x_var_eps
    
    # Reshape into original shape
    x_norm = x_norm.reshape(N, C, H, W)

    # Scale and shift the normalised data using gamma and beta
    out = x_norm * gamma + beta

    cache = x_mean_centre, x_var_eps, x_norm, gamma, g_shape

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Initialise variables
    x_mean_centre, x_var_eps, x_norm, gamma, g_shape = cache
    N, C, H, W = dout.shape
    
    # Compute gradients
    dgamma = np.sum(dout * x_norm, axis=(0,2,3), keepdims=True) # 1xCx1x1, sum over (N, H, W)
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True) # 1xCx1x1, sum over (N, H, W)
    dnorm = dout * gamma
    
    # Similar to forward pass, reshape X into (N',D'), where N' = N*G and 
    # D' = (C/G)*H*W dimensions. Then re-use LayerNorm's backward pass.
    dnorm = dnorm.reshape(g_shape)
    x_norm = x_norm.reshape(g_shape)
    N_, D_ = g_shape
    
    # From batchnorm_backward_alt derivations, adjust:
    # axis=0 -> axis=1 to sum over D' for each sample instead of N'
    # Replacing N' -> D' to reflect the number of features D' instead of N'
    dx = ((dnorm * D_) - x_norm * np.sum(dnorm * x_norm, axis=1).reshape(N_, -1) - np.sum(dnorm, axis=1).reshape(N_, -1))/(D_ * x_var_eps)
    
    # Reshape to original shape
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
