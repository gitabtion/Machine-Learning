import random

import tensorflow as tf
import numpy as np
import cv2
import os

boat_train_patches_path = "../../../mcm_prime/data/train/patches/boat/"
water_train_patches_path = "../../../mcm_prime/data/train/patches/water/"
boat_test_patches_path = "../../../mcm_prime/data/test/boat/"
water_test_patches_path = "../../../mcm_prime/data/test/water/"
boat_train_files = os.listdir(boat_train_patches_path)
boat_test_files = os.listdir(boat_test_patches_path)
water_train_files = os.listdir(water_train_patches_path)
water_test_files = os.listdir(water_test_patches_path)

pic_height = 28
pic_width = 28
pic_size = pic_height * pic_width
pic_class = 2
n_channels = 3

n_inputs = pic_size
n_outputs = pic_class

n_filters_conv1 = 32
n_filters_conv2 = 64
n_inputs_full1 = 7 * 7 * n_filters_conv2
n_neurons_full1 = 1024
n_inputs_full2 = n_neurons_full1
n_neurons_full2 = n_outputs

filter_height = 5
filter_width = 5

n_train_batches = 20000
n_test_batches = 4000

n_epochs = 5


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


def bias_varibale(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_poo_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


xs = tf.placeholder(tf.float32, [None, pic_height, pic_width, n_channels])
ys = tf.placeholder(tf.float32, [None, pic_class])
# keep_prob = tf.placeholder(tf.float32)

# x_image = tf.reshape(xs, [-1, pic_height, pic_width, n_channels])

W_conv1 = weight_variable([filter_height, filter_width, n_channels, n_filters_conv1])
b_conv1 = bias_varibale([n_filters_conv1])

h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1)
h_pool = max_poo_2x2(h_conv1)

W_conv2 = weight_variable([filter_height, filter_width, n_filters_conv1, n_filters_conv2])
b_conv2 = bias_varibale([n_filters_conv2])

h_conv2 = tf.nn.relu(conv2d(h_pool, W_conv2) + b_conv2)
h_pool2 = max_poo_2x2(h_conv2)

# fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, n_inputs_full1])

W_fc1 = weight_variable([n_inputs_full1, n_neurons_full1])
b_fc1 = bias_varibale([n_neurons_full1])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# h_fc1_dropt = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([n_inputs_full2, n_neurons_full2])
b_fc2 = bias_varibale([n_neurons_full2])

prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
print(prediction.shape)
# 损失函数
# 交叉熵损失
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction)
loss = tf.reduce_mean(cross_entropy)

# 最优化算法
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

print(tf.argmax(prediction, 1).shape)
# 测试指标
correct_predtion = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predtion, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./save/mcm_prime.ckpt")
    for epoch in range(n_epochs):

        train_acc = .0
        test_acc = .0
        for batch in range(n_train_batches):
            x = random.randint(0, 20000)
            if batch % 2 == 0:
                x_batch = np.array([cv2.imread(water_train_patches_path + water_train_files[batch])])
                y_batch = np.array([[0, 1]])
            else:
                x_batch = np.array([cv2.imread(boat_train_patches_path + boat_train_files[batch])])
                y_batch = np.array([[1, 0]])
            _, acc = sess.run([train_op, accuracy], feed_dict={xs: x_batch, ys: y_batch})
            train_acc += acc
            print(acc)

        train_acc /= n_train_batches

        for batch in range(n_test_batches):
            if batch % 2 == 0:
                x_batch = np.array([cv2.imread(water_test_patches_path + water_test_files[batch])])
                y_batch = np.array([[0, 1]])
            else:
                x_batch = np.array([cv2.imread(boat_test_patches_path + boat_test_files[batch])])
                y_batch = np.array([[1, 0]])
            acc = sess.run(accuracy, feed_dict={xs: x_batch, ys: y_batch})
            test_acc += acc
        test_acc /= n_test_batches
        print("Epoch " + str(epoch) + ": Test_acc " + str(test_acc) + "train_acc: " + str(train_acc))
        saver.save(sess, "./save/mcm_prime.ckpt")
