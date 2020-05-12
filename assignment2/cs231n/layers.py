from builtins import range
import numpy as np

#affine_forward 위한 함수
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

    N = x.shape[0] #입력값 크기 지정
    D = w.shape[0] #가중치 크기 지정
    M = w.shape[1]
    out = x.reshape(N,D).dot(w) + b #output값은 wx+b의 형태로 계산된 값
    #차원 맞춰서 행렬곱(dot)으로 계산한다
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache

#backward 함수 정의
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

    #각 입력값 크기 저장, input값과 가중치 크기 정의
    N = x.shape[0]
    dx = np.dot(dout, w.T).reshape(x.shape) #x에 대한 편미분값
    dw = np.dot(x.reshape(N, -1).T, dout) #w에 대한 편미분값
    db = dout.T.sum(axis = 1) #bias에 대한 편미분값

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


#forward할때 앞서 구한 값을 activation function에 넣어 output값을 구한다
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

    #0보다 작은 값은 0이 되는 relu 특성
    out = np.maximum(0, x)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache

#activation function 거친 값에 대해 backward할 대 미분값구하는 함수
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

    #relu 미분하면 0보다 작을 때 0, 0보다 클때 1이다
    dx = dout*(x>0)
    
    #x에 대한 기울기값은 relu미분값 * 앞서 구한값이다(chain rule)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


#batch normalization에 대한 foward pass 함수 정의
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
-> 이공식 사용!

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

    #batch normalization은 입력층의 입력 데이터는 쉽게 normalization할 수 있지만, 입력층을 지나서
    #만나게 되는 레이어들의 입력은 normalization하기 쉽지 않다. 
    #batch normalization은 이러한 문제들을 해결하기 위한 알고리즘!

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
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


        mu = np.mean(x, axis = 0) #먼저 평균과

        xmu = x- mu
        sq = xmu ** 2
        var = np.var(x, axis = 0) #variance값을 구한다

        #batch normalization은 mini-batch단위로 데이터의 분포가 평균이 0,분산이 1이 되도록
        #정규화 한다

        running_mean = momentum * running_mean + (1-momentum) *mu
        running_var = momentum * running_var + (1-momentum)*var
        #이동평균법을 사용해서 평균과 표준편차를 계산한다
        #이러한 running mean과 running var은 학습단계에서 매 minibatch마다 업데이트한다

        sqrtvar = np.sqrt(var + eps)
        ivar = 1./sqrtvar
        xhat = xmu * ivar

        out = gamma * xhat + beta

        cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

        #정규화를 하였을 때 대부분이 output값이 0에 가까워진다
        #이때 sigmoid와 같은 활성화 함수의 입력값으로 들어가면 선형구간에 빠져버리는 문제가 발생하는데
        #이를 해결하기 위해 scaling과 shifting을 해주는 gamma와 beta를 적용하여 계산한다

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_normalize = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalize + beta

        #test의 겨우에도 running_mean과 running_var을 이용하여 normalization을 진행하고
        #scaling과 shifting을 위한 gamma와 beta를 이용하여 계산한다

        pass


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
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
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  
    #batchnormalization에 대한 backporpagation과정

    N, D = dout.shape

    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

    #batch normalization에 대한 파라미터 였던 gamma와 beta에 대한 미분값을 구해야한다
    #batch normalization에 대한 미분값을 구하는 식을 통해 x에 대한 미분값도 구한다
 
    #http://sanghyukchun.github.io/88/ 이거 참고하면서 함

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout*xhat, axis=0)

    dxhat = dout * gamma
    divar = np.sum(dxhat*xmu, axis=0)

    dxmu1 = dxhat * ivar 

    dsqrtvar = -1. / (sqrtvar**2) * divar
    dvar = 0.5 * 1. / np.sqrt(var+eps) * dsqrtvar
    dsq = 1. / N * np.ones((N, D)) * dvar

    dxmu2 = 2 * xmu * dsq 

    dx1 = dxmu1 + dxmu2

    dmu = -1 * np.sum(dx1, axis=0)

    dx2 = 1. / N * np.ones((N, D)) * dmu

    dx = dx1 + dx2 
    pass

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

     #위의 함수와 동일하다
    #각 미분값들을 간단한 식으로 표현하여 구한다
    N, D = dout.shape
    
    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

    dxhat = dout * gamma

    #앞의 함수와 다르게 mu와 var, std를 output에 대해 바로 편미분한 값으로
    #식에 대입하여 dx를 구한다
    dx = 1.0/N * ivar * (N*dxhat - np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat, axis=0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xhat*dout, axis=0)
   
    pass

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
    eps = ln_param.get('eps', 1e-5)
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

    #앞서 설명되어 있는것 처럼
    #layer normalization은 batch normalization과 달리 
    #batch normalization 전에 gamma와 beta 파라미터에 의해 동일하게 scale된다 
    
    # Tranpose x (layernormalization위해) ->(D, N)
    x = x.T

	  # 그다음은 batch normalization과 동일하게 진행함
    mu = np.mean(x, axis=0)

    xmu = x - mu
    sq = xmu ** 2
    var = np.var(x, axis=0)

    sqrtvar = np.sqrt(var + eps)
    ivar = 1./sqrtvar
    xhat = xmu * ivar

    # 다시 (N,D)로
    xhat = xhat.T
 
    out = gamma * xhat + beta

    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    pass

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

    #layer normalization foward과정으로 구한 각 값들을 먼저 정의하고
    

    #각각에 대한 기울기 값들을 구한다
    #x에 대한 기울기값을 구하기 위해 mu, var, std를 gamma에 대해 미분한 값들을 구하고
    # 그 값들을 이용해서 x미분값을 구한다
    #아까 사용했던 batchnormalization_alt의 코드와 동일하게 흘러간다

    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xhat*dout, axis=0)

    dxhat = dout * gamma

    # Transpose 
    dxhat = dxhat.T
    xhat = xhat.T # (N,D)로 다시 back

    N, D = xhat.shape

    #dx계산
    dx = 1.0/N * ivar * (N*dxhat - np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat, axis=0))

    # Transpose dx back
    dx = dx.T
    pass

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
    p, mode = dropout_param['p'], dropout_param['mode'] #dopout를 위한 파라미터값 p를 선정
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #먼저 train의 경우

        mask = (np.random.rand(*x.shape)< p) / p
        #dropout은 random으로 그 비율만큼 제거시킨다 

        out = x*mask #dropout

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #test의 경우

        out = x #test는 dropout을 시키지 않고 그대로 진행
        pass

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
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #dropout의 backpropagation과정
        #cache에 저장되어있는 mask값을 곱하여 dx를 구함.
        
        dx = dout* mask
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


#assignment 2-4 part
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

    #필요한 parameter들 정의
    stride = conv_param['stride']
    pad = conv_param['pad']
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    
    #다음 convolutional layer로 들어가기 위한 값 계산
    H_n = int(1 + (H + 2 * pad - HH) / stride)
    W_n = int(1 + (W + 2 * pad - WW) / stride)
    X_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant', constant_values=0)
    out = np.zeros((N,F,H_n,W_n))

    #forward 진행 
    for i in range(N):
        for j in range(H_n):
            for k in range(W_n):
                for f in range(F):
                    X_i = X_pad[i]
                    inp_con = X_i[:,j*stride:j*stride+HH,k*stride:k*stride+WW]
                    out_con = (inp_con*w[f,:,:,:]).sum() + b[f]
                    out[i,f,j,k] = out_con
  
    pass

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

    #필요한 파라미터 정의
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    #padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    H_n = 1 + (H + 2 * pad - HH) // stride
    W_n = 1 + (W + 2 * pad - WW) // stride

    #각각 gradient 구함
    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    #backpropagation 진행
    for n in range(N):
        for f in range(F):
            db[f] += dout[n, f].sum()
            for j in range(0, H_n):
                for i in range(0, W_n):
                    dw[f] += x_pad[n,:,j*stride:j*stride+HH,i*stride:i*stride+WW]*dout[n,f,j,i]
                    dx_pad[n,:,j*stride:j*stride+HH,i*stride:i*stride+WW] += w[f]*dout[n, f, j, i]
    dx = dx_pad[:,:,pad:pad+H,pad:pad+W]
    pass

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

    #필요한 파라미터 정의
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    #maxpooling이 포함될 때 공식을 이용
    #이 때는 no padding is necessary
    H_n = int(1 + (H - pool_height) / stride)
    W_n = int(1 + (W - pool_width) / stride)

    out = np.zeros((N,C,H_n,W_n))
    
    #forward 진행
    for i in range(N):
        for j in range(H_n):
            for k in range(W_n):
                for l in range(C):
                    x_max = x[i,l,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width]
                    out[i,l,j,k] = np.amax(x_max) #가장 큰 값 반환

    pass
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

    #필요한 정의 정의
    x,pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    #앞서 forward에서 사용되었던 공식과 동일
    H_n = int(1 + (H - pool_height) / stride)
    W_n = int(1 + (W - pool_width) / stride)

    #backpropagation 진행 -> dx구함
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(H_n):
            for k in range(W_n):
                for l in range(C):
                    index = np.argmax(x[i,l,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width]) # 가장 큰 값 인덱스 반환
                    ind1,ind2 = np.unravel_index(index,(pool_height, pool_width)) #좌표배열로 변환

                    dx[i,l,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width][ind1,ind2] = dout[i,l,j,k] #for문 돌리면서 각각 gradient 구함

    pass

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
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    _,C,_,_ = x.shape
    running_mean = bn_param.get('running_mean', np.zeros((1,C,1,1), dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros((1,C,1,1), dtype=x.dtype))

    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #batch normalization은 train과 test 적용방식이 다름 
    if mode == 'train':
      #평균, 분산 구하고
      #gamma, beta로 scaling, shifting 
        mu = np.mean(x,axis = (0,2,3),keepdims = True) #평균
        num = x-mu
        square_mu = num**2
        var = np.mean(square_mu,axis = (0,2,3),keepdims = True) #분산
        sqrtvar = np.sqrt(var + eps)
        inverse_var = 1/sqrtvar 
        norm = num*inverse_var #normalization

        gamma = gamma.reshape(1,C,1,1)
        beta = beta.reshape(1,C,1,1)
        scale_norm = gamma*norm
        shift_norm = scale_norm + beta
        out = shift_norm #batch norm train 끝!

        #이동평균 저장 -> test시 사용함

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

        cache = (beta,gamma,norm,num,var,eps,sqrtvar)
        
    elif mode == 'test':
        out_hat = (x-running_mean)/np.sqrt(running_var+eps) #train시 구한 이동평균 사용
        out = gamma.reshape(1,C,1,1)*out_hat + beta.reshape(1,C,1,1)
    
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    pass

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

    beta,gamma,norm,num,var,eps,sqrtvar = cache
    #각각의 gradient구함
    #bath normalization의 backpropagation chain rule 부분 사용
    N,C,H,W = dout.shape
    dbeta = np.sum(dout,axis = (0,2,3))
    dgamma = np.sum(dout*norm,axis = (0,2,3))

    dmu = np.mean(dout, axis=(0,2,3),keepdims = True) 
    dvar = 2 * np.mean(num*dout, axis=(0,2,3),keepdims = True)
    dstd= dvar/(2*sqrtvar)
    dx = gamma.reshape(1,C,1,1)*((dout - dmu)*sqrtvar - dstd*(num))/sqrtvar**2
    pass

    pass

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
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #group normalization
    #G: integer number of groups to split into, should be a divisor of C
    N,C,H,W = x.shape
    x = x.reshape(N,G,C//G,H,W)
    #평균, 분사 구함
    mu = np.mean(x,axis = (2,3,4),keepdims = True)
    num = x-mu
    square_mu = num**2
    var = np.mean(square_mu,axis = (2,3,4),keepdims = True)
    sqrtvar = np.sqrt(var + eps)

    inverse_var = 1/sqrtvar 
    norm = num*inverse_var
    norm = norm.reshape(N,C,H,W)
    
    #scale, shift위한 gamma,beta 설정
    gamma = gamma.reshape(1,C,1,1) #C로 reshape
    beta = beta.reshape(1,C,1,1)
    scale_norm = gamma*norm
    shift_norm = scale_norm + beta
    out = shift_norm


    cache = (beta,gamma,norm,num,var,eps,sqrtvar,G)


    pass

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

    beta,gamma,norm,num,var,eps,sqrtvar,G = cache
    #batch normalization backpropagation
    N,C,H,W = dout.shape

    #각각의 gradient
    dbeta = np.sum(dout,axis = (0,2,3),keepdims=True)
    dgamma = np.sum(dout*norm,axis = (0,2,3),keepdims=True)

    dout = (gamma*dout).reshape(N,G,C//G,H,W)
    
    dmu = np.mean(dout, axis=(2,3,4),keepdims = True) 
    dvar = 2 * np.mean(num*dout, axis=(2,3,4),keepdims = True)
    dstd= dvar/(2*sqrtvar)
    
    dx_before = ((dout - dmu)*sqrtvar - dstd*(num))/sqrtvar**2
    dx = dx_before.reshape(N,C,H,W) #reshape

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


#svm classification을 이용한 loss와 gradient구하기
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
    #svm특성이 margin구하기
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N #margin의 합을 N으로 나눈 평균으로 loss값을 구한다
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx

#softmax classification을 이용한 loss와 gradient구하기
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
    loss = -np.sum(log_probs[np.arange(N), y]) / N #softmax 함수를 이용하여 
    #loss를 구한다 
    dx = probs.copy()
    dx[np.arange(N), y] -= 1 #미분한값(기울기도) 구한다
    dx /= N
    return loss, dx
