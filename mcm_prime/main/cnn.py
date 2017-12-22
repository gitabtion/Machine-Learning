import tensorflow as tf
import numpy as np
import cv2
import os

boat_train_patches_path = "../data/train/patches/boat/"
water_train_patches_path = "../data/train/patches/water/"
boat_test_patches_path = "../data/test/boat/"
water_test_patches_path = "../data/test/water/"
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
n_neurons_full1 = 96
n_inputs_full2 = n_neurons_full1
n_neurons_full2 = n_outputs

filter_height = 5
filter_width = 5

batch_size = 100

n_train_batches = 20000
n_test_batches = 4000

n_epochs = 5


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


x = tf.placeholder(tf.float32, [pic_width, pic_height, n_channels])
y = tf.placeholder(tf.float32, [n_outputs])
# keep_prob = tf.placeholder(tf.float32)

# 定义神经网络模型
#
# mnist.train.images.shape = [55000,28*28]
# [-1,28,28,3]
# [-1,pic_height,pic_width,n_channels]

x_images = tf.reshape(x, [-1, pic_height, pic_width, n_channels])

# 第一层卷积神经网络的权值和偏置值
W_conv1 = weight([filter_height, filter_width, n_channels, n_filters_conv1])
b_conv1 = bias([n_filters_conv1])
# 卷积
sigma_conv1 = conv2d(x_images, W_conv1) + b_conv1
# 非线性变换
relu_conv1 = tf.nn.relu(sigma_conv1)

# 池化
pool_conv1 = max_pool_2x2(relu_conv1)

# 第二层卷积神经网络的权值和偏置值
W_conv2 = weight([filter_height, filter_width, n_filters_conv1, n_filters_conv2])
b_conv2 = bias([n_filters_conv2])

# 卷积
sigma_conv2 = conv2d(pool_conv1, W_conv2) + b_conv2
# 非线性变换
relu_conv2 = tf.nn.relu(sigma_conv2)

# 池化
pool_conv2 = max_pool_2x2(relu_conv2)

# 把卷积网络的输出数据格式化为全连接层所要求的格式
pool_flat_conv2 = tf.reshape(pool_conv2, [-1, n_inputs_full1])

# 第一次全连接层
W_full1 = weight([n_inputs_full1, n_neurons_full1])
b_full1 = bias([n_neurons_full1])
# 加法
sigma_full1 = tf.matmul(pool_flat_conv2, W_full1) + b_full1

# 非线性变换
tanh_full1 = tf.nn.tanh(sigma_full1)

#
# keep_full1 = tf.(tanh_full1, keep_prob)

# 第二次全连接层
W_full2 = weight([n_neurons_full1, n_neurons_full2])
b_full2 = bias([n_neurons_full2])
# 加法
sigma_full2 = tf.matmul(tanh_full1, W_full2) + b_full2

# 非线性变换
softmax_full2 = tf.nn.softmax(sigma_full2)

prediction = softmax_full2

# 损失函数
# 交叉熵损失
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)
loss = tf.reduce_mean(cross_entropy)

# 最优化算法
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

# 测试指标
correct_predtion = tf.equal(tf.argmax(y, 0), tf.argmax(prediction, 0))
accuracy = tf.reduce_mean(tf.cast(correct_predtion, tf.float32))

# 建立会话进行训练和测试

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    # sess.run(init)
    saver.restore(sess,"./mcm_prime.ckpt")
    # 迭代训练
    for epoch in range(n_epochs):

        train_acc = .0
        test_acc = .0
        for batch in range(n_train_batches):
            if (batch % 1000 == 0):
                saver.save(sess, "./mcm_prime.ckpt")
            if batch % 2 == 0:
                x_batch = cv2.imread(water_train_patches_path + water_train_files[batch])
                y_batch = [0, 1]
            else:
                x_batch = cv2.imread(boat_train_patches_path + boat_train_files[batch])
                y_batch = [1, 0]
            _, acc = sess.run([train_op, accuracy], feed_dict={x: x_batch, y: y_batch})
            train_acc += acc

        train_acc /= n_train_batches

        for batch in range(n_test_batches):
            if batch % 2 == 0:
                x_batch = cv2.imread(water_test_patches_path + water_test_files[batch])
                y_batch = [0, 1]
            else:
                x_batch = cv2.imread(boat_test_patches_path + boat_test_files[batch])
                y_batch = [1, 0]
            acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
            test_acc += acc
        test_acc /= n_test_batches
        print("Epoch " + str(epoch) + ": Test_acc " + str(test_acc) + "train_acc: " + str(train_acc))
    saver.save(sess, "./mcm_prime.ckpt")