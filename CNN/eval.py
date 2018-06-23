import tensorflow as tf
import numpy as np
import datetime
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Data
tf.flags.DEFINE_string('mnist_dir', '/Users/abtion/workspace/dataset/mnist', 'dir of mnist data')

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1529737570/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
#
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

print("\nEvaluating...\n")


def eval():
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    print(checkpoint_file)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout/dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            cnn_loss = graph.get_operation_by_name('loss/loss').outputs[0]
            cnn_acc = graph.get_operation_by_name('accuracy/accuracy').outputs[0]

            mnist = input_data.read_data_sets(FLAGS.mnist_dir, one_hot=True)
            n_test_batches = int(np.ceil(mnist.test.num_examples))
            steps = []
            accs = []
            for batch in range(n_test_batches):
                x_batch, y_batch = mnist.test.next_batch(FLAGS.batch_size)
                loss, acc = sess.run([cnn_loss, cnn_acc],
                                     {input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0})
                time_str = datetime.datetime.now().isoformat()
                steps.append(batch)
                accs.append(acc)
                print("{}:\tloss {:g},\tacc {:g}".format(time_str, loss, acc))
                if batch % 100 == 99:
                    plt.plot(steps, accs, 'o')
                    plt.show()


if __name__ == '__main__':
    eval()
