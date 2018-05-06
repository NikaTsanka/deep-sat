#! /usr/bin/env python

'''
CNN-DeepSat-4

Members: David Ng Wu, Nika Tsankashvili and Jamila Vargas
04/03/2018
High Performance Machine Learning
Prof. Jianting Zhang

'''
import os
import time
import numpy as np
import scipy.io
import tensorflow as tf
from layers import conv_layer, max_pool_2x2, full_layer

cwd = os.getcwd()
DATA_PATH = ''.join( [cwd, '/', 'sat-4-full.mat' ] )
BATCH_SIZE = 128
STEPS = 21875 #epoch 5
lr = 0.0001
#epoch = steps * batchsize / totalbatch
# decay = 1e-9  # .00000001
decay = 0.9  # .00000001
momentum = 0
dropoutProb = 0.5


def run_simple_net():
    dataset = DeepSatData()

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 4])
    y_ = tf.placeholder(tf.float32, shape=[None, 4])
    keep_prob = tf.placeholder(tf.float32)

    conv1 = conv_layer(x, shape=[3, 3, 4, 16], pad='SAME')
    conv1_pool = avg_pool_2x2(conv1, 2, 2) #28x28x4->14x14x16

    conv2 = conv_layer(conv1_pool, shape=[3, 3, 16, 32], pad='SAME')
    conv2_pool = avg_pool_2x2(conv2, 2, 2)  #14x14x16->7x7x32

    conv3 = conv_layer(conv2_pool, shape=[3, 3, 32, 64], pad='SAME')
    # conv3_pool = max_pool_2x2(conv3) # 7x7x32 ->7x7x64

    conv4 = conv_layer(conv3, shape=[3, 3, 64, 96], pad='SAME')
    # conv4_pool = max_pool_2x2(conv4) # 7x7x64 -> 7x7x96

    conv5 = conv_layer(conv4, shape=[3, 3, 96, 64], pad='SAME')
    conv5_pool = avg_pool_2x2(conv5, 2, 2) # 7x7x96 ->7x7x64

    _flat = tf.reshape(conv5_pool, [-1, 3 * 3 * 64])
    _drop1 = tf.nn.dropout(_flat, keep_prob=keep_prob)

    # full_1 = tf.nn.relu(full_layer(_drop1, 200))
    full_1 = tf.nn.relu(full_layer(_drop1, 512))
    # -- until here
    # classifier:add(nn.Threshold(0, 1e-6))
    _drop2 = tf.nn.dropout(full_1, keep_prob=keep_prob)
    full_2 = tf.nn.relu(full_layer(_drop2, 256))
    # classifier:add(nn.Threshold(0, 1e-6))
    full_3 = full_layer(full_2, 4)

    predict = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=full_3, labels=y_))

    #train_step = tf.train.RMSPropOptimizer(lr, decay, momentum).minimize(predict)
    train_step = tf.train.AdamOptimizer(lr).minimize(predict)

    correct_prediction = tf.equal(tf.argmax(full_3, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def test(sess):
        X = dataset.test.images.reshape(10, 10000, 28, 28, 4)
        Y = dataset.test.labels.reshape(10, 10000, 4)
        acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0})
                       for i in range(10)])
        return acc


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sum_writer = tf.summary.FileWriter('logs/' + 'default')
        # sum_writer.add_graph(sess.graph)

        for i in range(STEPS):
            batch = dataset.train.random_batch(BATCH_SIZE)
            #batch = dataset.train.next_batch(BATCH_SIZE)
            batch_x = batch[0]
            batch_y = batch[1]

            # _, summ = sess.run([train_step, merged_sum], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            # sum_writer.add_summary(summ, i)

            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropoutProb})

            epoch = 4000
            if i % epoch == 0:
                print("\n*****************EPOCH: %d" % (i/epoch))
            if i % 500 == 0:
                acc = test(sess)
                loss = sess.run(predict, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropoutProb})
                print("EPOCH:%d" % (i/epoch) + " Step:" + str(i) + "|| Minibatch Loss= " + "{:.4f}".format(loss) + " Accuracy: {:.4}%".format(acc * 100))

        test(sess)
        # sum_writer.close()


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
        print(self.images.shape)
        print(self.labels.shape)
        return self

    def next_batch(self, batch_size):
        x = self.images[self._i:self._i+batch_size]
        y = self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y




    def random_batch(self, batch_size):
        n = len(self.images)
        ix = np.random.choice(n, batch_size)
        return self.images[ix], self.labels[ix]


class DeepSatData:
    def __init__(self):
        self.train = DeepSatLoader('train').load_data()
        self.test  = DeepSatLoader('test').load_data()


def load_mat_data():
    import matplotlib.pyplot as plt

    data = DeepSatData()

    print(data.train.images.shape)
    print(data.train.labels.shape)

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


if __name__ == "__main__":
    start_time = time.time()
    run_simple_net()
    # load_mat_data()
    print("--- %s seconds ---" % (time.time() - start_time))


'''
EPOCH:0 Step:0|| Minibatch Loss= 2.2798 Accuracy: 19.91%
EPOCH:0 Step:500|| Minibatch Loss= 1.2444 Accuracy: 78.1%
EPOCH:0 Step:1000|| Minibatch Loss= 2.0128 Accuracy: 64.53%
EPOCH:0 Step:1500|| Minibatch Loss= 1.0333 Accuracy: 68.45%
EPOCH:0 Step:2000|| Minibatch Loss= 1.6406 Accuracy: 80.06%
EPOCH:0 Step:2500|| Minibatch Loss= 2.4697 Accuracy: 72.31%
EPOCH:0 Step:3000|| Minibatch Loss= 8.3938 Accuracy: 41.15%
EPOCH:0 Step:3500|| Minibatch Loss= 1.1109 Accuracy: 54.4%

*****************EPOCH: 1.000000
EPOCH:1 Step:4000|| Minibatch Loss= 1.7280 Accuracy: 47.84%
EPOCH:1 Step:4500|| Minibatch Loss= 1.5185 Accuracy: 54.12%
EPOCH:1 Step:5000|| Minibatch Loss= 1.8099 Accuracy: 59.69%
EPOCH:1 Step:5500|| Minibatch Loss= 7.0275 Accuracy: 61.6%
EPOCH:1 Step:6000|| Minibatch Loss= 1.4996 Accuracy: 60.57%
EPOCH:1 Step:6500|| Minibatch Loss= 0.9451 Accuracy: 71.82%
EPOCH:1 Step:7000|| Minibatch Loss= 0.8016 Accuracy: 76.3%
EPOCH:1 Step:7500|| Minibatch Loss= 0.7202 Accuracy: 66.16%

*****************EPOCH: 2.000000
EPOCH:2 Step:8000|| Minibatch Loss= 2.6838 Accuracy: 73.15%
EPOCH:2 Step:8500|| Minibatch Loss= 2.1093 Accuracy: 63.92%
EPOCH:2 Step:9000|| Minibatch Loss= 1.1112 Accuracy: 50.81%
EPOCH:2 Step:9500|| Minibatch Loss= 0.8932 Accuracy: 68.44%
EPOCH:2 Step:10000|| Minibatch Loss= 2.8521 Accuracy: 73.36%
EPOCH:2 Step:10500|| Minibatch Loss= 5.3190 Accuracy: 67.18%
EPOCH:2 Step:11000|| Minibatch Loss= 0.8163 Accuracy: 65.5%
EPOCH:2 Step:11500|| Minibatch Loss= 0.7703 Accuracy: 64.11%

*****************EPOCH: 3.000000
EPOCH:3 Step:12000|| Minibatch Loss= 1.2966 Accuracy: 52.27%
EPOCH:3 Step:12500|| Minibatch Loss= 2.6385 Accuracy: 51.35%
EPOCH:3 Step:13000|| Minibatch Loss= 6.4430 Accuracy: 58.46%
EPOCH:3 Step:13500|| Minibatch Loss= 1.1491 Accuracy: 48.93%
EPOCH:3 Step:14000|| Minibatch Loss= 1.5351 Accuracy: 35.63%
EPOCH:3 Step:14500|| Minibatch Loss= 2.1135 Accuracy: 75.01%
EPOCH:3 Step:15000|| Minibatch Loss= 1.7280 Accuracy: 35.63%
EPOCH:3 Step:15500|| Minibatch Loss= 1.3266 Accuracy: 36.99%

*****************EPOCH: 4.000000
EPOCH:4 Step:16000|| Minibatch Loss= 3.8214 Accuracy: 50.56%
EPOCH:4 Step:16500|| Minibatch Loss= 1.4057 Accuracy: 35.63%
EPOCH:4 Step:17000|| Minibatch Loss= 1.0046 Accuracy: 49.34%
EPOCH:4 Step:17500|| Minibatch Loss= 3.1696 Accuracy: 35.63%
EPOCH:4 Step:18000|| Minibatch Loss= 1.3685 Accuracy: 35.63%
EPOCH:4 Step:18500|| Minibatch Loss= 1.2900 Accuracy: 35.63%
EPOCH:4 Step:19000|| Minibatch Loss= 1.3276 Accuracy: 35.63%
EPOCH:4 Step:19500|| Minibatch Loss= 1.3353 Accuracy: 35.63%

*****************EPOCH: 5.000000
EPOCH:5 Step:20000|| Minibatch Loss= 1.3300 Accuracy: 35.63%
EPOCH:5 Step:20500|| Minibatch Loss= 1.3951 Accuracy: 35.63%
EPOCH:5 Step:21000|| Minibatch Loss= 1.3387 Accuracy: 35.63%
EPOCH:5 Step:21500|| Minibatch Loss= 1.3441 Accuracy: 35.63%
EPOCH:5 Step:22000|| Minibatch Loss= 3.8251 Accuracy: 53.93%
EPOCH:5 Step:22500|| Minibatch Loss= 1.1427 Accuracy: 45.5%
EPOCH:5 Step:23000|| Minibatch Loss= 1.1344 Accuracy: 46.36%
EPOCH:5 Step:23500|| Minibatch Loss= 1.3571 Accuracy: 35.63%

*****************EPOCH: 6.000000
EPOCH:6 Step:24000|| Minibatch Loss= 2.1620 Accuracy: 45.43%
EPOCH:6 Step:24500|| Minibatch Loss= 1.4117 Accuracy: 35.63%
EPOCH:6 Step:25000|| Minibatch Loss= 1.3388 Accuracy: 35.63%
EPOCH:6 Step:25500|| Minibatch Loss= 1.1946 Accuracy: 49.74%
EPOCH:6 Step:26000|| Minibatch Loss= 1.1722 Accuracy: 46.03%
EPOCH:6 Step:26500|| Minibatch Loss= 1.3075 Accuracy: 35.63%
EPOCH:6 Step:27000|| Minibatch Loss= 1.3128 Accuracy: 35.63%
EPOCH:6 Step:27500|| Minibatch Loss= 1.1162 Accuracy: 50.95%

*****************EPOCH: 7.000000
EPOCH:7 Step:28000|| Minibatch Loss= 1.1261 Accuracy: 48.41%
EPOCH:7 Step:28500|| Minibatch Loss= 1.3456 Accuracy: 53.03%
EPOCH:7 Step:29000|| Minibatch Loss= 1.0420 Accuracy: 47.65%
EPOCH:7 Step:29500|| Minibatch Loss= 1.8819 Accuracy: 71.4%
EPOCH:7 Step:30000|| Minibatch Loss= 0.9264 Accuracy: 76.26%
EPOCH:7 Step:30500|| Minibatch Loss= 1.2366 Accuracy: 47.0%
EPOCH:7 Step:31000|| Minibatch Loss= 2.2979 Accuracy: 52.41%
EPOCH:7 Step:31500|| Minibatch Loss= 0.6190 Accuracy: 74.2%

*****************EPOCH: 8.000000
EPOCH:8 Step:32000|| Minibatch Loss= 5.6514 Accuracy: 51.42%
EPOCH:8 Step:32500|| Minibatch Loss= 18.7082 Accuracy: 35.63%
EPOCH:8 Step:33000|| Minibatch Loss= 1.3056 Accuracy: 36.25%
EPOCH:8 Step:33500|| Minibatch Loss= 1.3405 Accuracy: 35.63%
EPOCH:8 Step:34000|| Minibatch Loss= 1.3540 Accuracy: 35.63%
EPOCH:8 Step:34500|| Minibatch Loss= 1.2898 Accuracy: 35.63%
EPOCH:8 Step:35000|| Minibatch Loss= 1.3265 Accuracy: 35.63%
EPOCH:8 Step:35500|| Minibatch Loss= 1.3353 Accuracy: 35.63%

*****************EPOCH: 9.000000
EPOCH:9 Step:36000|| Minibatch Loss= 1.3398 Accuracy: 35.63%
EPOCH:9 Step:36500|| Minibatch Loss= 1.3949 Accuracy: 35.63%
EPOCH:9 Step:37000|| Minibatch Loss= 1.3387 Accuracy: 35.63%
EPOCH:9 Step:37500|| Minibatch Loss= 1.3306 Accuracy: 35.63%
EPOCH:9 Step:38000|| Minibatch Loss= 1.3701 Accuracy: 35.63%
EPOCH:9 Step:38500|| Minibatch Loss= 1.2582 Accuracy: 35.63%
EPOCH:9 Step:39000|| Minibatch Loss= 1.3272 Accuracy: 35.63%
EPOCH:9 Step:39500|| Minibatch Loss= 1.3351 Accuracy: 35.63%

*****************EPOCH: 10.000000
EPOCH:10 Step:40000|| Minibatch Loss= 1.3398 Accuracy: 35.63%
EPOCH:10 Step:40500|| Minibatch Loss= 1.3951 Accuracy: 35.63%
EPOCH:10 Step:41000|| Minibatch Loss= 1.3387 Accuracy: 35.63%
EPOCH:10 Step:41500|| Minibatch Loss= 1.3440 Accuracy: 35.63%
EPOCH:10 Step:42000|| Minibatch Loss= 1.3702 Accuracy: 35.63%
EPOCH:10 Step:42500|| Minibatch Loss= 1.2901 Accuracy: 35.63%
EPOCH:10 Step:43000|| Minibatch Loss= 1.3276 Accuracy: 35.63%
EPOCH:10 Step:43500|| Minibatch Loss= 1.3352 Accuracy: 35.63%

*****************EPOCH: 11.000000
EPOCH:11 Step:44000|| Minibatch Loss= 1.3398 Accuracy: 35.63%
EPOCH:11 Step:44500|| Minibatch Loss= 1.3948 Accuracy: 35.63%
EPOCH:11 Step:45000|| Minibatch Loss= 1.3066 Accuracy: 35.63%

'''
