from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


#two layer로 이루어진 neuralnet 정의
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    #먼저 초기화
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #twolayer의 가중치와 bias 초기화
        self.params['W1'] = np.random.randn(input_dim, hidden_dim)*weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes)*weight_scale
        self.params['b2'] = np.zeros(num_classes)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    #loss function 정의
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        a,c1 = affine_forward(X,self.params['W1'], self.params['b1']) 
        #layers.py에서 정의했던 forward function을 이용하여 첫번째 layer을 거친다

        a_relu, c1_relu = relu_forward(a) #그 다음 activation function을 거치고

        scores, c2 = affine_forward(a_relu, self.params['W2'], self.params['b2'])
        #두번째 layer역시 forward로 거쳐 최종 output을 얻는다 
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #forward propagation과정으로 얻은 기울기값들을 이용해 backpropagation을 진행하고
        #이를 이용하여 가중치를 업데이트 시킨다

        #backpropagation하는 과정 code

        N = X.shape[0]

        #먼저 activation에 대해 편미분값을 구한다
        scores = np.exp(scores)
        correct_class_score = scores[range(N), y].reshape(-1,1)
        
        #loss를 구하는 식으로 loss를 구한다
        loss = -np.sum(np.log(correct_class_score/np.sum(scores,axis=1).reshape(-1,1)))/N
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])))

        dev = scores/np.sum(scores,axis =1).reshape(-1,1)
        dev[range(N),y]-=1

        grad_a_relu, grads['W2'], grads['b2'] = affine_backward(dev,c2) #그를 이용하여 
        #layers.py에서 정의한 back propagation 과정의 함수에 넣는다
        #이를 통해 각 가중치, activation function relu, bias에 대한 기울기 값을 얻는다

        #이제는 그 값들을 기반으로 가중치를 업데이트 시킨다
        #더 작은 loss로 줄이기 위해 가중치를 업데이트 시키는 과정인데
        #앞서 구한 편미분값에 기존의 가중치를 곱하는 형식으로 업데이트 된다

        grads['W2']/= N
        grads['W2'] += self.reg*self.params['W2']
        grads['b2']/= N

        grad_a = relu_backward(grad_a_relu, c1_relu) #activation function backward를 거치고

        #첫번째 layer역시 앞서 한 것과 동일한 방식으로 gradient를 기반으로 
        #가중치를 update시킨다

        grad_x, grads['W1'], grads['b1'] = affine_backward(grad_a, c1)
        grads['W1']/= N
        grads['W1'] += self.reg*self.params['W1']
        grads['b1']/= N

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

#fullyconnectednets에 대한 클래스 함수이다
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    #먼저 필요한 것들 초기화

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #파라미터를 초기화하고, 모든 값들을 저장한다
        #normalization으로 평균0, 분산1에 머물도록 scaling한다(batch normalization 진행)

 #fullconnectedlyaer의 특징은 hidden layer의 수를 임의적으로 정한다
        #그러므로 어떤 수가 들어오느냐에 따라 파라미터 수가 달라지므로
        #for문을 사용하여 layer 수에 따라 파라미터를 지정할 수 있도록 한다

        for i in range(self.num_layers):
            W_i = 'W' + str(i+1)
            b_i = 'b' + str(i+1)

            # Output layer (without bath normalization)
            if i == self.num_layers - 1:
                self.params[W_i] = np.random.randn(hidden_dims[len(hidden_dims)-1],
                    num_classes) * weight_scale
                self.params[b_i] = np.zeros(num_classes)
            
            # With batch normalization
            else:
                # 1 hidden layer
                if i == 0:
                    self.params[W_i] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
                    self.params[b_i] = np.zeros(hidden_dims[0])
                # 중간 hidden layer
                else:
                    self.params[W_i] = np.random.randn(hidden_dims[i-1], hidden_dims[i]) * weight_scale
                    self.params[b_i] = np.zeros(hidden_dims[i])

                # Batch normalization 또는 layer normalization 각각 parameter 저장
                if self.normalization in ['batchnorm', 'layernorm']:
                    self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
                    self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i])
        

          #batch normalization을 하도록 지정하였을 때
          #gamma값과 beta값을 구해야 한다 그에 대한 if문도 넣는다


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.

        #dropout에 대한 파라미터도 지정
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.ln_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            
    #loss function도 정의
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #먼저 forward pass 진행한다
        
        caches = {}

        for i in range(self.num_layers-1):
            W_i = 'W' + str(i+1)
            b_i = 'b' + str(i+1)

            # 첫번째 hidden layer
            if i == 0:
                out = X

            # batch normalization의 경우
            if self.normalization == 'batchnorm':
                fc_out, fc_cache = affine_forward(out, self.params[W_i], self.params[b_i])
                bn_out, bn_cache = batchnorm_forward(fc_out, self.params['gamma'+str(i+1)],
                    self.params['beta'+str(i+1)], self.bn_params[i])
                out, relu_cache = relu_forward(bn_out)
                #affine -> batch norm -> relu
                caches[i+1] = (fc_cache, bn_cache, relu_cache)

            # layer normalization의 경우
            elif self.normalization == 'layernorm':
                fc_out, fc_cache = affine_forward(out, self.params[W_i], self.params[b_i])
                ln_out, ln_cache = layernorm_forward(fc_out, self.params['gamma'+str(i+1)],
                    self.params['beta'+str(i+1)], self.ln_params[i])
                out, relu_cache = relu_forward(ln_out)
                # affine -> layer norm -> relu
                caches[i+1] = (fc_cache, ln_cache, relu_cache)

            # batch normalization, layer normalization 
            else:
                out, caches[i+1] = affine_relu_forward(out, self.params[W_i],
                    self.params[b_i])

            # Dropout
            if self.use_dropout:
                out, caches['dropout'+str(i+1)] = dropout_forward(out, self.dropout_param)

        # 마지막부분(최종)
        scores, caches[self.num_layers] = affine_forward(out,
            self.params['W'+str(self.num_layers)],
            self.params['b'+str(self.num_layers)])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #foward를 진행한 값을 기반으로 backpropagation역시 진행한다
        
        loss, dscores = softmax_loss(scores, y) #softmax loss결과

        for i in range(self.num_layers, 0, -1):

            # 식으로 계산
            loss += 0.5 * self.reg * np.sum(np.square(self.params['W'+str(i)]))

            # 마지막 hiddenlayer부터 반대로 진행
            if i == self.num_layers:
                dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dscores, caches[i])
            else:
                # Dropout 부터 
                if self.use_dropout:
                    dout = dropout_backward(dout, caches['dropout'+str(i)])

                 # drelu -> dbatchnorm/ dlayernorm -> daffine으로 backpropagation진행

                if self.normalization == 'batchnorm':
                   
                    fc_cache, bn_cache, relu_cache = caches[i]
                    dbn_out = relu_backward(dout, relu_cache)
                    #각각 backward function으로 gradient구한다
                    dfc_out, grads['gamma'+str(i)], grads['beta'+str(i)] = batchnorm_backward(dbn_out, bn_cache)
                    dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dfc_out, fc_cache)
                
                elif self.normalization == 'layernorm':
                    fc_cache, ln_cache, relu_cache = caches[i]
                    dln_out = relu_backward(dout, relu_cache)
                     #각각 backward function으로 gradient구한다
                    dfc_out, grads['gamma'+str(i)], grads['beta'+str(i)] = layernorm_backward(dln_out, ln_cache)
                    dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dfc_out, fc_cache)

                else:
                  #relu까지
                    dout, grads['W'+str(i)], grads['b'+str(i)] = affine_relu_backward(dout, caches[i])
            
            # Regularization term
            grads['W'+str(i)] += self.reg * self.params['W'+str(i)]
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
