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
    out = x.reshape(x.shape[0],w.shape[0]).dot(w) + b
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

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #dout->dloss/dz,dloss/dx=dout*df/dx<->用dx表示
    #function=xw+b
    #dx=dout*wT 在backward作为dout向前传递
    #dw=xT*dout
    #db=dout
    dx=dout.dot(w.T).reshape(x.shape)
    dw=x.reshape(dout.shape[0],w.shape[0]).T.dot(dout)
    db=np.sum(dout,axis=0).T
    #axis=0沿着行方向计算每列总和，返回一个一维行数组
    #axis=1沿着列方向计算每行总和，返回一个一维列数组
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
    out = np.maximum(0,x)
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
    dx=dout
    dx[x<=0]=0
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

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
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None#tuple
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        sample_mean=x.mean(axis=0)
        sample_var=x.var(axis=0)
       
        normalized_x = (x - sample_mean) / np.sqrt(sample_var + eps)
        out=gamma*normalized_x+beta
        
        running_mean=momentum*running_mean+(1-momentum)*sample_mean
        running_var=momentum*running_var+(1-momentum)*sample_var
        cache=(x,gamma,beta,eps,sample_mean,sample_var)      
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        normalized_x = (x - running_mean) / np.sqrt(running_var + eps)
        out=gamma*normalized_x+beta
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

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
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    x,gamma,beta,eps,sample_mean,sample_var=cache
    N,D=x.shape
    
    normalized_x=(x - sample_mean) / np.sqrt(sample_var + eps)
    
    dgamma=np.sum(dout,axis=0)
    dbeta=np.sum(dout*normalized_x,axis=0)
    
    dx_hat=dout*gamma
    dvar=np.sum(dx_hat*(x-sample_mean)*(-0.5)*(sample_var+eps)**(-1.5),axis=0)
    dmean=np.sum(dx_hat*(-1)/np.sqrt(sample_var+eps)+dvar*(-2)/N*np.sum(x-sample_mean,axis=0))
    
    dx=dx_hat*1/np.sqrt(sample_var + eps)+dvar * 2 * (x - sample_mean) / N + dmean / N
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
    x,gamma,beta,eps,mean,var=cache
    norm=(x - mean) / np.sqrt(var + eps)
    dgamma=np.sum(dout,axis=0)
    dbeta=np.sum(dout*norm,axis=0)
    a = x - mean
    b = var + eps
    m = x.shape[0]
    dx = (1. / m) * gamma * (b**(-1. / 2.)) * \
       (m * dout - dout.sum(axis=0) - a * b**(-1) * (dout * a).sum(axis=0))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout
        and rescale the outputs to have the same mean as at test time.
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        mask=np.random.rand(*x.shape)<p
        out=x*mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        out = x

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
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        dx=dout*mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (H + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape
    stride,pad=conv_param.values()
    
    H_out=1 + (H + 2 * pad - HH) // stride#取整数部分，不四舍五入
    W_out=1 + (H + 2 * pad - WW) //stride
    out=np.zeros((N, F, H_out , W_out))
    
    padding = ((0, 0),  # No padding for batch dimension
           (0, 0),  # No padding for channels dimension
           (pad, pad),  # Pad 1 row at the top and 1 row at the bottom
           (pad, pad)) # Pad 1 column on the left and 1 column on the right
    x_padded = np.pad(x, padding, mode='constant', constant_values=0)
    
    for n in range(N):
        for f in range(F):
            for h in range(H_out):
                for ww in range(W_out):
                    out[n, f, h, ww] = np.sum(x_padded[n, :, h*stride:h*stride+HH, ww*stride:ww*stride+WW] * w[f, :]) + b[f]

    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
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
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    x, w, b, conv_param=cache
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape
    _,_,H_out,W_out=dout.shape
    stride,pad=conv_param.values()
    
    padding = ((0, 0),  # No padding for batch dimension
           (0, 0),  # No padding for channels dimension
           (pad, pad),  # Pad 1 row at the top and 1 row at the bottom
           (pad, pad)) # Pad 1 column on the left and 1 column on the right
    x_padded = np.pad(x, padding, mode='constant', constant_values=0)
    
    db=np.sum(dout,axis=(0,2,3))
    dw=np.zeros((F,C,HH,WW))#dw =kernel of x*dout per mask
    dx_=np.zeros_like(x_padded)#kernel of dx= w*dout per mask
    
    for n in range(N):
        for f in range(F):
            for h in range(H_out):
                for ww in range(W_out):
                    dw[f]+=x_padded[n,:,h*stride:h*stride+HH,ww*stride:ww*stride+WW]*dout[n,f,h,ww]
                    dx_[n,:,h*stride:h*stride+HH,ww*stride:ww*stride+WW]+=w[f]*dout[n,f,h,ww]
    dx=dx_[:,:,pad:H+pad,pad:W+pad]
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    N, C, H, W= x.shape
    pool_height,pool_width,stride=pool_param.values()
    H_out=1+(H-pool_height)//stride
    W_out=1+(W-pool_width)//stride
    out=np.zeros((N,C,H_out,W_out))
    
 
    for h in range(H_out):
        for w in range(W_out):
            out[:,:,h,w]=np.max(\
     x[:,:,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width],axis=(2,3))
         
                                
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param=cache
    N, C, H, W= x.shape
    pool_height,pool_width,stride=pool_param.values()
    H_out=1+(H-pool_height)//stride
    W_out=1+(W-pool_width)//stride
    dx=np.zeros_like(x)
 
    for h in range(H_out):
        for w in range(W_out):
            x_slice=x[:,:,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width]
            '''
            max_indices=np.argmax(x_slice,axis=(2,3))
            #argmax返回最大值索引(C, N, 1, 1)的四维数组
            dx_indices = (np.arange(N)[:, None, None], np.arange(C)[:, None, None], max_indices // pool_width, max_indices % pool_width)
             
            dout_slice=dout[:,:,h,w][:,:,none,none]
            #将原始的二维梯度切片 (C, N) 转换为 (C, N, 1, 1) 的四维数组
            dx[dx_indices] += dout_slice
            #np.arange(N) 创建一个包含从0到N-1的整数的一维数组
           #[:, None, None] 添加两个新的维度，形状 (N, 1, 1)。
        
        #dx_indices 的四个数组用于确定每个最大值在输入数据 dx 中的位置，
        #其中第一个数组确定样本数
        #第二个数组确定通道数
        #第三个数组确定最大值在池化窗口中的行索引
        #第四个数组确定最大值在池化窗口中的列索引。
        '''
            mask=(x_slice==np.max(x_slice,axis=(2,3))[:,:,None,None])
            dx[:,:,h*stride:h*stride+pool_height,w*stride:w*stride+pool_width]+=\
            (dout[:,:,h,w])[:,:,None,None]*mask
            #add a 2x3 array to a 2x1 array, the result will be a 2x3 array
            #dimension will broadcasting automatically
            
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
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

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W=x.shape
    x_transform=x.transpose(0,3,2,1).reshape(N*W*H,C)
    out,cache=batchnorm_forward(x_transform,gamma,beta,bn_param)
    out=out.reshape(N,W,H,C).transpose(0,3,2,1)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

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

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W=dout.shape
    dout_transform=dout.transpose(0,3,2,1).reshape(N*W*H,C)
    dx,dgamma,dbeta=batchnorm_backward(dout_transform,cache)
    dx=dx.reshape(N,W,H,C).transpose(0,3,2,1)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
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
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    #减去最大值提高稳定性并计算指数（归一化前）
    probs /= np.sum(probs, axis=1, keepdims=True)
    #x中的值变成概率（归一化后）
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    #probs[np.arange(N), y])锁定真实标签对应的概率
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    #对每个样本的真实类别处的概率减去了1
    dx /= N
    #每个样本的平均梯度
    return loss, dx
