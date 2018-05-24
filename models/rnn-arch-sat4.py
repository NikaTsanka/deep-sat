'''
Group Project - Code

Members: David Ng Wu, Nika Tsankashvili and Jamila Vargas
05/24/2018
High Performance Machine Learning
Prof. Jianting Zhang

'''

import time
import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


# Training Parameters
DATA_PATH = '/home/nikatsanka/Workspace/tensor-env/deep-sat-datasets/sat-4-full.mat'
training_steps = 15625  # 15625
batch_size = 128
display_step = 200
lr = 0.001
decay = 0.9

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28). -
time_steps = 28  # time_steps. to h^n propagate
num_hidden = 128  # hidden layer num of features
num_classes = 4  # MNIST total classes (0-9 digits) LABELS


class DeepSatLoader:
    def __init__(self, key):
        self._key = key
        self._i = 0
        self.images = None
        self.labels = None

    def load_data(self):
        data = scipy.io.loadmat(DATA_PATH)
        self.images = data[self._key + '_x'].transpose(3, 0, 1, 2).astype(float) / 255
        self.labels = data[self._key + '_y'].transpose(1, 0)
        return self

    def next_batch(self, batch_size):
        x = self.images[self._i:self._i+batch_size]
        y = self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y


class DeepSatData:
    def __init__(self):
        self.train = DeepSatLoader('train').load_data()
        self.test  = DeepSatLoader('test').load_data()


def rnn_model_trainer():
    # tf Graph input
    X = tf.placeholder(tf.float32, [None, time_steps, num_input], name='X')  # 128, 28, 28
    Y = tf.placeholder(tf.int32, [None, num_classes], name='Y')  # 128, 10

    Wl = tf.Variable(tf.truncated_normal([num_hidden, num_classes],  mean=0, stddev=.01))
    bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))

    cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)  # 128 | 128, 28, 28
    last_rnn_output = outputs[:, -1, :]
    logits = tf.matmul(last_rnn_output, Wl) + bl

    pred = tf.nn.softmax(logits, name='pred')
    loss_op = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y)

    # train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_op)
    train_op = tf.train.AdamOptimizer(lr).minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # for tensorboard
    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar('accuracy', accuracy)
    merged_sum = tf.summary.merge_all()

    data = DeepSatData()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sum_writer = tf.summary.FileWriter('logs/' + '1d-input-5-epochs')
    sum_writer.add_graph(sess.graph)

    # training loop
    for step in range(1, training_steps + 1):
        batch_x, batch_y = data.train.next_batch(batch_size)
        # gray-scale transformation
        batch_x = np.dot((batch_x[:, :, :, :])[..., :3], [0.299, 0.587, 0.114])
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # For Tensorboard
        _, summ = sess.run([train_op, merged_sum], feed_dict={X: batch_x, Y: batch_y})
        sum_writer.add_summary(summ, step)

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Mini_batch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Training Finished!")
    test_images = data.test.images[:batch_size]
    test_images = np.dot((test_images[:, :, :, :])[..., :3], [0.299, 0.587, 0.114])
    test_label = data.test.labels[:batch_size]
    test_acc = sess.run(accuracy, feed_dict={X: test_images, Y: test_label})
    print("Final Testing Accuracy:", test_acc)

    sum_writer.close()


if __name__ == "__main__":
    start_time = time.time()
    rnn_model_trainer()
    print("--- %s seconds ---" % (time.time() - start_time))
