# based on https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
# and https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/blob/master/05__text_and_visualizations/BasicRNNCell.py

import time
import scipy.io
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

# Training Parameters
DATA_PATH = '/home/nikatsanka/Workspace/tensor-env/deep-sat-datasets/sat-4-full.mat'
training_steps = 5000
batch_size = 128
display_step = 200
lr = 0.001
decay = 0.9

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28). -
time_steps = 28  # time_steps. to h^n propagate
num_hidden = 128  # hidden layer num of features
num_classes = 4  # MNIST total classes (0-9 digits) LABELS
num_channels = 4


def run_rnn_net():
    # tf Graph input
    X = tf.placeholder(tf.float32, [None, time_steps, num_input, num_channels], name='X')  # 28, 28
    Y = tf.placeholder(tf.int32, [None, num_classes], name='Y')  # 10

    # variation 1: initializing W1 and b1
    # Wl = tf.Variable(tf.random_normal([num_hidden, num_classes]))  # shape = 128, 10 (WR) weight
    # bl = tf.Variable(tf.random_normal([num_classes]))  # 10 labels
    Wl = tf.Variable(tf.truncated_normal([num_hidden, num_classes],  mean=0, stddev=.01))
    bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))

    # variation 2: choosing RNN cell
    # tanh activation
    # with tf.name_scope('HCell'):
    # cell = rnn.BasicRNNCell(num_hidden)  # 128
    cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)


    '''
    Internally, tf.nn.rnn creates an unrolled graph for a fixed RNN length. 
    That means, if you call tf.nn.rnn with inputs having 200 time steps you 
    are creating a static graph with 200 RNN steps.
    tf.nn.dynamic_rnn solves this. It uses a 
    tf.While loop to dynamically construct the graph when it is executed. 
    '''
    # vairation 3: choosing static/dynamic rnn
    # unpack 28 tensors from x(28,28) values by chipping it along the 1 axis?
    # https://medium.com/machine-learning-algorithms/build-basic-rnn-cell-with-static-rnn-707f41d31ee1
    # https://www.dotnetperls.com/stack-tensorflow
    # http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
    # X_T = tf.unstack(X, time_steps, 1)  # one slice, one peak/look at the image. first row
    # outputs, states = tf.nn.static_rnn(cell, X_T, dtype=tf.float32)
    # logits = tf.matmul(outputs[-1], Wl) + bl  # ?????????????????
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    last_rnn_output = outputs[:, -1, :]
    logits = tf.matmul(last_rnn_output, Wl) + bl

    prediction = tf.nn.softmax(logits)
    loss_op = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y)

    # variation 4: choose optimizer
    # train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_op)
    train_op = tf.train.RMSPropOptimizer(lr, decay).minimize(loss_op)

    # Evaluate model
    # https://www.tensorflow.org/tutorials/layers#logits_layer
    # https://stackoverflow.com/questions/41708572/tensorflow-questions-regarding-tf-argmax-and-tf-equal
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # for tensorboard
    # tf.summary.scalar('loss', loss_op)
    # tf.summary.scalar('accuracy', accuracy)
    # merged_sum = tf.summary.merge_all()
    # start_time = time.time()

    # initialization
    # open file here
    # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    data = DeepSatData()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # hparam = 'V3'

    # sum_writer = tf.summary.FileWriter('logs/' + hparam)
    # sum_writer.add_graph(sess.graph)

    # training loop
    for step in range(1, training_steps + 1):
        batch_x, batch_y = data.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, time_steps, num_input))  # 128, 28, 28
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # _, summ = sess.run([train_op, merged_sum], feed_dict={X: batch_x, Y: batch_y})

        # sum_writer.add_summary(summ, step)

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Mini_batch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            # tf.summary.scalar('accuracy', acc)
            # sum_writer.add_summary(acc, step)

    print("Training Finished!")
    # print("--- %s seconds ---" % (time.time() - start_time))

    # testing using a batch
    test_data = data.test.images[:batch_size].reshape(-1, time_steps, num_input)
    test_label = data.test.labels[:batch_size]
    test_acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
    print("Final Testing Accuracy:", test_acc)

    # merged_sum = tf.summary.merge_all()

    # sum_writer.close()


class DeepSatLoader:
    def __init__(self, key):
        self._key = key
        self._i = 0
        self.images = None
        self.labels = None

    def load_data(self):
        data = scipy.io.loadmat(DATA_PATH)
        # print(data['annotations'])
        self.images = data[self._key + '_x'].transpose(3, 0, 1, 2).astype(float) / 255
        self.labels = data[self._key + '_y'].transpose(1, 0)
        # print(self.images.shape)
        # print(self.labels.shape)
        return self

    def next_batch(self, batch_size):
        # ellipsis (…) to make a selection tuple of the
        # same length as the dimension of an array.
        # x = self.images[..., self._i:self._i+batch_size]
        # y = self.labels[..., self._i:self._i+batch_size]
        x = self.images[self._i:self._i+batch_size]
        y = self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y


class DeepSatData:
    def __init__(self):
        self.train = DeepSatLoader('train').load_data()
        self.test  = DeepSatLoader('test').load_data()


def load_mat_data_test():
    # import matplotlib.pyplot as plt

    '''
    train
    (400000, 28, 28, 4)
    (400000, 4)
    test
    (100000, 28, 28, 4)
    (100000, 4)
    '''

    data = DeepSatData()
    print('train')
    print(data.train.images.shape)
    print(data.train.labels.shape)
    print('test')
    print(data.test.images.shape)
    print(data.test.labels.shape)



    # for i in range(4):
    #     batch = data.train.next_batch(100)
    #     print(batch[0].shape)
    #     print(batch[1].shape)
    # ind = 5
    # batch = data.train.next_batch(ind)
    # print(batch[0][0,0,0,0])
    # print(type(batch[0][0,0,0,0]))

    # plt.figure()
    # im = []
    # for i in range(ind):
    #     im.append(batch[0][:,:,:,i])
    #     print(batch[1][:,i])
    # # print(im.shape)
    # for i in range(ind):
    #     plt.imshow(im[i])
    #     plt.show()

    # plt.imshow(im)
    # plt.show()
    # display_cifar(batch[0], 28)

    '''
    (28, 28, 4, 400000)
    (4, 400000)
    (28, 28, 4, 100000)
    (4, 100000)
    (4, 2)
    '''
    # dataset = scipy.io.loadmat(DATA_PATH)
    # train_x = dataset['train_x']
    # train_y = dataset['train_y']
    #
    # test_x = dataset['test_x']
    # test_y = dataset['test_y']
    #
    # labels = dataset['annotations']
    #
    # # print(type(train_x))
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)
    #
    # print(labels.shape)

    # for i in dataset.values():
    #     print(len(i))

    # print(type(dataset['train_y']))


def load_mnist_data_test():
    # https://www.tensorflow.org/versions/r1.2/get_started/mnist/mechanics
    '''
    data_sets.train	        55000 images and labels, for primary training.
    data_sets.validation	5000 images and labels, for iterative validation of training accuracy.
    data_sets.test	        10000 images and labels, for final testing of trained accuracy.

    before reshape train
    (55000, 784)
    (55000, 10)
    before reshape test
    (10000, 784)
    (10000, 10)
    after reshape train batch
    (128, 28, 28)
    (128, 10)
    testing:
    (128, 28, 28)
    (128, 10)
    '''
    data = input_data.read_data_sets("/tmp/data/", one_hot=True)
    print('before reshape train')
    print(data.train.images.shape)
    print(data.train.labels.shape)
    print('before reshape test')
    print(data.test.images.shape)
    print(data.test.labels.shape)
    print('after reshape train batch')
    batch_x, batch_y = data.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, time_steps, num_input))  # 128, 28, 28
    print(batch_x.shape)
    print(batch_y.shape)

    print('testing:')
    test_data = data.test.images[:batch_size].reshape(-1, time_steps, num_input)
    test_label = data.test.labels[:batch_size]
    print(test_data.shape)
    print(test_label.shape)


if __name__ == "__main__":
    start_time = time.time()
    run_rnn_net()
    # load_mat_data_test()
    # load_mnist_data_test()
    print("--- %s seconds ---" % (time.time() - start_time))
