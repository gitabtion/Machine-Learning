import tensorflow as tf
import numpy as np


class CNN(object):
    """
    A CNN classifier of mnist.
    """

    def __init__(self, height=28, width=28, classes=10, filter_width=5, filter_height=5, n_channels=1, batch_size=100, n_epochs=1):
        self.n_filters_conv1 = 32
        self.n_filters_conv2 = 64
        self.n_inputs_full1 = 7*7*self.n_filters_conv2
        self.n_neurons_fc1 = 1024

        self.input_x = tf.placeholder(
            tf.float32, [None, height*width], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, classes], name='input_y')

        with tf.name_scope('reshape'):
            x_image = tf.reshape(self.input_x, [-1, height, width, n_channels])

        with tf.name_scope('conv1'):
            W_conv1 = _weight([filter_height, filter_width,
                               n_channels, self.n_filters_conv1])
            b_conv1 = _bias([n_filters_conv1])
            h_conv1 = tf.nn.relu(_conv2d(x_image, W_conv1)+b_conv1)

        with tf.name_scope('pool1'):
            h_pool1 = _max_pool_2x2(h_conv1)

        with tf.name_scope('conv2'):
            W_conv2 = _weight(
                [filter_height, self.n_filters_conv1, self.n_filters_conv2])
            b_conv2 = _bias([n_filters_conv2])
            h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2)+b_conv2)

        with tf.name_scope('pool2'):
            h_pool2 = _max_pool_2x2(h_conv2)

        with tf.name_scope('fc1'):
            W_fc1 = _weight([self.n_inputs_full1, self.n_neurons_fc1])
            b_fc1 = _bias([self.n_neurons_fc1])

            h_pool2_flat = tf.reshape(h_pool2, [-1, self.n_inputs_full1])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('fc2'):
            W_fc2 = _weight([self.n_neurons_fc1, classes])
            b_fc2 = _bias([classes])

            self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    def _weight(shape):
        init = tf.truncated_normal(shape, stddev=2)
        return tf.Variable(init)

    def _bias(shape):
        init = tf.constant(.0, shape=shape)
        return tf.Variable(init)

    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
