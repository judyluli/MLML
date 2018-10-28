"""Supporting functions for arbitrary order Factorization Machines."""

import math
import numpy as np
import tensorflow as tf
import itertools
from itertools import combinations_with_replacement, takewhile, count
from collections import defaultdict

def batch_norm(x, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """

    with tf.variable_scope('bn'):

        scale = tf.Variable(tf.ones([x.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([x.get_shape()[-1]]))
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, scale, 1e-3)
        #normed = tf.nn.batch_normalization(x, mean, var, None, None, 1e-3)
    return normed

def matmul_wrapper(A, B, optype):
    """Wrapper for handling sparse and dense versions of `tf.matmul` operation.

    Parameters
    ----------
    A : tf.Tensor
    B : tf.Tensor
    optype : str, {'dense', 'sparse'}

    Returns
    -------
    tf.Tensor
    """
    with tf.name_scope('matmul_wrapper') as scope:
        if optype == 'dense':
            return tf.matmul(A, B)
        elif optype == 'sparse':
            return tf.sparse_tensor_dense_matmul(A, B)
        else:
            raise NameError('Unknown input type in matmul_wrapper')


def pow_wrapper(X, p, optype):
    """Wrapper for handling sparse and dense versions of `tf.pow` operation.

    Parameters
    ----------
    X : tf.Tensor
    p : int
    optype : str, {'dense', 'sparse'}

    Returns
    -------
    tf.Tensor
    """
    with tf.name_scope('pow_wrapper') as scope:
        if optype == 'dense':
            return tf.pow(X, p)
        elif optype == 'sparse':
            return tf.SparseTensor(X.indices, tf.pow(X.values, p), X.dense_shape)
        else:
            raise NameError('Unknown input type in pow_wrapper')


def count_nonzero_wrapper(X, optype):
    """Wrapper for handling sparse and dense versions of `tf.count_nonzero`.

    Parameters
    ----------
    X : tf.Tensor (N, K)
    optype : str, {'dense', 'sparse'}

    Returns
    -------
    tf.Tensor (1,K)
    """
    with tf.name_scope('count_nonzero_wrapper') as scope:
        if optype == 'dense':
            return tf.count_nonzero(X, axis=0, keep_dims=True)
        elif optype == 'sparse':
            indicator_X = tf.SparseTensor(X.indices, tf.ones_like(X.values), X.dense_shape)
            return tf.sparse_reduce_sum(indicator_X, axis=0, keep_dims=True)
        else:
            raise NameError('Unknown input type in count_nonzero_wrapper')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Predefined loss functions
# Should take 2 tf.Ops: outputs, targets and should return tf.Op of element-wise losses
# Be careful about dimensionality -- maybe tf.transpose(outputs) is needed

def loss_logistic(outputs, y):
    margins = -y * tf.transpose(outputs)
    raw_loss = tf.log(tf.add(1.0, tf.exp(margins)))
    return tf.minimum(raw_loss, 100, name='truncated_log_loss')

def loss_mse(outputs, y):
    return tf.pow(y -  tf.transpose(outputs), 2, name='mse_loss')

