import tensorflow as tf
import numpy as np

def conv2d(input, output_dim=64, kernel=(5, 5), strides=(2, 2), stddev=0.02, name='conv_2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('Conv2dW', [*kernel,input.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Conv2db', [output_dim], initializer=tf.zeros_initializer())

        return tf.nn.conv2d(input, w, strides=[1, *strides, 1], padding='SAME') + b

def dconv2d(input, output_dim, batch_size, kernel=(5, 5), strides=(2, 2), stddev=0.02, name='deconv_2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('Deconv2dW', [*kernel, output_dim, input.get_shape()[-1]],initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Deconv2db', [output_dim], initializer=tf.zeros_initializer())
        input_shape  = input.get_shape().as_list()
        output_shape = [batch_size, int(input_shape[1] * strides[0]), int(input_shape[2] * strides[1]),output_dim]
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=[1, *strides, 1])
    
        return deconv + b

def dense(input, output_dim, stddev=0.02, name='dense'):
    with tf.variable_scope(name):
        shape = input.get_shape()
        w = tf.get_variable('DenseW', [shape[1], output_dim], tf.float32,tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('Denseb', [output_dim],initializer=tf.zeros_initializer())
        
        return tf.matmul(input, w) + b

def batch_norm(input, name='bn'):
    
    with tf.variable_scope(name):
        output_dim = input.get_shape()[-1]
        beta = tf.get_variable('BnBeta', [output_dim],initializer=tf.zeros_initializer())
        gamma = tf.get_variable('BnGamma', [output_dim],initializer=tf.ones_initializer())
        if len(input.get_shape()) == 2:
            mean, var = tf.nn.moments(input, [0])
        else:
            mean, var = tf.nn.moments(input, [0, 1, 2])
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)
    
def leaky_relu(input, leak=0.2, name='lrelu'):
    return tf.nn.leaky_relu(input, leak)
