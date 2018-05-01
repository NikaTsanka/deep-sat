import time
import numpy as np
import scipy.io
import tensorflow as tf
from models.layers import conv_layer, max_pool_2x2, full_layer


DATA_PATH = '/home/nikatsanka/Workspace/tensor-env/deep-sat-datasets/sat-4-full.mat'
# HYPERS
NUM_SAMPLES = 400000
EPOCHS = 5
BATCH_SIZE = 128
STEPS = int((NUM_SAMPLES * EPOCHS) / BATCH_SIZE)
ONE_EPOCH = int(NUM_SAMPLES / BATCH_SIZE)
TEST_INTERVAL = BATCH_SIZE * 5
MODELS_TO_KEEP = 5
lr = 0.0001
decay = 0.9
momentum = 0
dropoutProb = 0.5

log_name = 'V1-CURRENT'

print('HYPER_PARAMETERS_USED')
print('---------------------')
print('NUM_SAMPLES:', NUM_SAMPLES)
print('EPOCHS:', EPOCHS)
print('BATCH_SIZE:', BATCH_SIZE)
print('STEPS:', STEPS)
print('LEARNING_RATE:', lr)
print('DECAY:', decay)
print('MOMENTUM:', momentum)
print('DROPOUT:', dropoutProb)


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7


def run_simple_net():
    dataset = DeepSatData()

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 4])
    y_ = tf.placeholder(tf.float32, shape=[None, 4])
    keep_prob = tf.placeholder(tf.float32)

    conv1 = conv_layer(x, shape=[3, 3, 4, 16], pad='VALID')
    conv1_pool = max_pool_2x2(conv1, 2, 2)

    conv2 = conv_layer(conv1_pool, shape=[3, 3, 16, 48], pad='SAME')
    conv2_pool = max_pool_2x2(conv2, 3, 3)

    conv3 = conv_layer(conv2_pool, shape=[3, 3, 48, 96], pad='SAME')
    # conv3_pool = max_pool_2x2(conv3)

    conv4 = conv_layer(conv3, shape=[3, 3, 96, 64], pad='SAME')
    # conv4_pool = max_pool_2x2(conv4)

    conv5 = conv_layer(conv4, shape=[3, 3, 64, 64], pad='SAME')
    conv5_pool = max_pool_2x2(conv5, 2, 2)

    _flat = tf.reshape(conv5_pool, [-1, 3 * 3 * 64])
    _drop1 = tf.nn.dropout(_flat, keep_prob=keep_prob)

    # full_1 = tf.nn.relu(full_layer(_drop1, 200))
    full_1 = tf.nn.relu(full_layer(_drop1, 200))
    # -- until here
    # classifier:add(nn.Threshold(0, 1e-6))
    _drop2 = tf.nn.dropout(full_1, keep_prob=keep_prob)
    full_2 = tf.nn.relu(full_layer(_drop2, 200))
    # classifier:add(nn.Threshold(0, 1e-6))
    full_3 = full_layer(full_2, 4)

    predict = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=full_3, labels=y_))

    train_step = tf.train.RMSPropOptimizer(lr, decay, momentum).minimize(predict)
    # train_step = tf.train.AdamOptimizer(lr)

    correct_prediction = tf.equal(tf.argmax(full_3, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', predict)
    tf.summary.scalar('accuracy', accuracy)
    merged_sum = tf.summary.merge_all()

    def test(test_sess):
        x_ = dataset.test.images.reshape(10, 10000, 28, 28, 4)
        y = dataset.test.labels.reshape(10, 10000, 4)
        test_acc = np.mean([test_sess.run(accuracy, feed_dict={x: x_[im], y_: y[im], keep_prob: 1.0})
                            for im in range(10)])
        return test_acc

    # config=config
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tensorboard
        sum_writer = tf.summary.FileWriter('logs/' + log_name)
        sum_writer.add_graph(sess.graph)

        # Create a Saver object
        saver = tf.train.Saver(max_to_keep=MODELS_TO_KEEP)

        for i in range(STEPS):
            batch = dataset.train.next_batch(BATCH_SIZE)
            batch_x = batch[0]
            batch_y = batch[1]

            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropoutProb})

            _, summ = sess.run([train_step, merged_sum], feed_dict={x: batch_x, y_: batch_y, keep_prob: dropoutProb})
            sum_writer.add_summary(summ, i)

            if i % ONE_EPOCH == 0:
                print("\n*****************EPOCH: %d" % ((i/ONE_EPOCH) + 1))
            if i % TEST_INTERVAL == 0:
                acc = test(sess)
                loss = sess.run(predict, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropoutProb})
                print("EPOCH:%d" % ((i/ONE_EPOCH) + 1) + " Step:" + str(i) + "|| Minibatch Loss= " +
                      "{:.4f}".format(loss) + " Accuracy: {:.4}%".format(acc * 100))
                # Create a checkpoint in every iteration
                saver.save(sess, './saved-models/cnn-model_iter',
                           global_step=i)

        test(sess)
        sum_writer.close()


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
        # print(self.images.shape)
        # print(self.labels.shape)
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


def load_mat_data():
    # import matplotlib.pyplot as plt

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
