'''
Group Project - Code

Members: David Ng Wu, Nika Tsankashvili and Jamila Vargas
05/24/2018
High Performance Machine Learning
Prof. Jianting Zhang

'''

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, p):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=p)


def max_pool_2x2(x, kw, kh):
    return tf.nn.max_pool(x, ksize=[1, kw, kh, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


def avg_pool_2x2(x, kw, kh):
    return tf.nn.avg_pool(x, ksize=[1, kw, kh, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


def conv_layer(input, shape, pad):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W, pad) + b)


def conv_layer_no_relu(input, shape, pad):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return conv2d(input, W, pad) + b


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    ret_val = tf.matmul(input, W) + b
    return ret_val
