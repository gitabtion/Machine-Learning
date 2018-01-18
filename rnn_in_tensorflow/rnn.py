import tensorflow as tf
from tensorflow.contrib.rnn import rnn_cell_impl
from tensorflow.contrib.rnn import rnn

sequence_length = 64
frame_size = 32
data = tf.placeholder(tf.float32, [None, sequence_length, frame_size])

num_neurons = 200
network = rnn_cell_impl.BasicRNNCell

outputs, states = rnn.dynamic_rnn(network, data, dtype=tf.float32)
