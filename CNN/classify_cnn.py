import tensorflow as tf
import numpy as np


class CNN(object):
    """
    A CNN classifier of mnist.
    """

    def __init__(self, height=28, width=28, classes=10, filter_height=5, filter_width=5, n_channels=1):
        self.n_filters_conv1 = 32
        self.n_filters_conv2 = 64
        self.n_inputs_full1 = 7 * 7 * self.n_filters_conv2
        self.n_neurons_fc1 = 1024

        self.input_x = tf.placeholder(
            tf.float32, [None, height * width], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, classes], name='input_y')

        with tf.name_scope('reshape'):
            x_image = tf.reshape(self.input_x, [-1, height, width, n_channels])

        with tf.name_scope('conv1'):
            W_conv1 = weight(shape=[filter_height, filter_width, n_channels, self.n_filters_conv1])
            b_conv1 = bias(shape=[self.n_filters_conv1])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('conv2'):
            W_conv2 = weight(shape=[filter_height,filter_width, self.n_filters_conv1, self.n_filters_conv2])
            b_conv2 = bias(shape=[self.n_filters_conv2])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)

        with tf.name_scope('fc1'):
            W_fc1 = weight(shape=[self.n_inputs_full1, self.n_neurons_fc1])
            b_fc1 = bias(shape=[self.n_neurons_fc1])

            h_pool2_flat = tf.reshape(h_pool2, [-1, self.n_inputs_full1])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        with tf.name_scope('output'):
            W_fc2 = weight(shape=[self.n_neurons_fc1, classes])
            b_fc2 = bias(shape=[classes])

            self.scores = tf.nn.xw_plus_b(h_fc1_drop, W_fc2, b_fc2, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses, name='loss')

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')


def weight(shape):
    init = tf.truncated_normal(shape, stddev=2)
    return tf.Variable(init)


def bias(shape):
    init = tf.constant(.0, shape=shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
