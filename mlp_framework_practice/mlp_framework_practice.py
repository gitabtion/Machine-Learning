import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

data_path = "./mnist/"
mnist = input_data.read_data_sets(data_path, one_hot=True)  # ont_hot coding

data_path2 = "./mnist_2/"


# mnist = input_data.read_data_sets(data_path) # raw_coding


# make this notebook's output stable across runs and clean default_graph
def reset_graph(seed=318):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()

# 图片属性
pic_height = 28
pic_width = 28
pic_size = pic_width * pic_height
pic_classes = 10
pic_channels = 1

# 网络拓扑
n_inputs = pic_size
n_hidden1 = 100
n_hidden2 = 50
n_outputs = pic_classes

# 迭代次数
n_epochs = 21

# mini-batch
batch_size = 10
n_train_batches = mnist.train.num_examples // batch_size
n_test_batches = mnist.test.num_examples // batch_size

# 学习率
learning_rate = 1e-4


#######################
#       构建模型       #
#######################

# 定义网络
def neurons_layer(X, n_neurons, activation=None):
    nn_inputs = int(X.get_shape()[-1])  # tf的返回值是tensor,取最后的一列
    stddev = 2 / np.sqrt(nn_inputs)
    init = tf.truncated_normal((nn_inputs, n_neurons), stddev=stddev)
    W = tf.Variable(init)
    b = tf.Variable(tf.zeros(n_neurons, tf.float32))
    sigma = tf.matmul(X, W) + b

    # 非线性变换
    if activation is not None:
        return activation(sigma)
    return sigma


#####################
#     构建神经网络    #
#####################

# 定义placeholder (batch_size ==> None)
X = tf.placeholder(tf.float32, (None, pic_size))
Y = tf.placeholder(tf.int64, None)
# Y = tf.placeholder(tf.float32,(None))
hidden1 = neurons_layer(X, n_hidden1, tf.nn.relu)  # (batch_size, n_hidden1)
hidden1_dropout = tf.nn.dropout(hidden1, 0.8)
hidden2 = neurons_layer(hidden1_dropout, n_hidden2, tf.nn.relu)  # (batch_size, n_hidden2)
hidden2_dropout = tf.nn.dropout(hidden2, 0.8)
prediction = neurons_layer(hidden2_dropout, n_outputs)  # (batch_size,n_outputs)

# 使用tf的框架
# hidden1 = tf.layers.dense(X, n_hidden1, tf.nn.relu)
# hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.relu)
# prediction = tf.layers.dense(hidden2, n_outputs)

# 交叉熵损失
xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction)
# xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=prediction)
loss = tf.reduce_mean(xentropy)

# 训练
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# 准确率
correct = tf.equal(tf.argmax(Y, 1), tf.argmax(prediction, 1))  # Y.shape()==(batch_size,pic_classes)
# correct = tf.nn.in_top_k(prediction, Y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 创建会话，训练、测试


# 保存模型
saver = tf.train.Saver()
model_path = "./my_model_mlp.ckpt"

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, model_path)

    # 多趟训练测试
    for epoch in range(n_epochs):

        # mini-batch
        train_acc = .0
        for batch in range(n_train_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            train_val, accuracy_value = sess.run([train, accuracy], feed_dict={X: x_batch, Y: y_batch})
            train_acc += accuracy_value
        train_acc /= n_train_batches

        test_acc = .0
        for batch in range(n_test_batches):
            x_batch, y_batch = mnist.test.next_batch(batch_size)
            accuracy_value = sess.run(accuracy, feed_dict={X: x_batch, Y: y_batch})
            test_acc += accuracy_value
        test_acc /= n_test_batches

        print("Epoch: " + str(epoch) + "\ntrain_acc: " + str(train_acc) + "\ntest_acc: " + str(test_acc))
        # saver.save(sess, model_path)
