import tensorflow as tf
import numpy as np
import os
import time
import datetime
from classify_cnn import CNN
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Parameters
# =========================================

# Data
tf.flags.DEFINE_string('mnist_dir', '/Users/abtion/workspace/dataset/mnist', 'dir of mnist data')

# Model Hyper parameters
tf.flags.DEFINE_integer('height', 28, 'Picture Height (default: 28)')
tf.flags.DEFINE_integer('width', 28, 'Picture Width (default: 28)')
tf.flags.DEFINE_integer('classes', 10, 'Picture Classes (default: 10)')
tf.flags.DEFINE_integer('filter_height', 5, 'Filter Height (default: 5)')
tf.flags.DEFINE_integer('filter_width', 10, 'Filter Width (default: 5)')
tf.flags.DEFINE_integer('n_channels', 1, 'Picture Channels (default: 1)')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 100, 'Batch Size (default: 100)')
tf.flags.DEFINE_integer('n_epochs', 100, 'Number of Training Epochs (default: 100)')
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1529721046/checkpoints", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def train():
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = CNN(
                height=FLAGS.height,
                width=FLAGS.width,
                classes=FLAGS.classes,
                filter_height=FLAGS.filter_height,
                filter_width=FLAGS.filter_width,
                n_channels=FLAGS.n_channels)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        print(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        print(checkpoint_prefix)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        steps = []
        accs = []

        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            steps.append(step)
            accs.append(accuracy)
            if step % 10 == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}:\tstep {},\tloss {:g},\tacc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        dev_steps = []
        dev_accs = []

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            dev_steps.append(step)
            dev_accs.append(accuracy)
            print("{}:\tstep {},\tloss {:g},\tacc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        mnist = input_data.read_data_sets(FLAGS.mnist_dir, one_hot=True)
        n_train_batches = int(np.ceil(mnist.train.num_examples))
        for epoch in range(FLAGS.n_epochs):
            for batch in range(n_train_batches):
                x_batch, y_batch = mnist.train.next_batch(FLAGS.batch_size)

                if batch == 0:
                    print("\nEvaluation:")
                    dev_step(x_batch, y_batch, writer=dev_summary_writer)
                    print("")

                train_step(x_batch, y_batch)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_batch, y_batch, writer=dev_summary_writer)
                    print("")

                    plt.plot(steps, accs, 'k', dev_steps, dev_accs, 'b')
                    plt.savefig('images/' + str(current_step) + '.png')
                    plt.show()
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
    train()
