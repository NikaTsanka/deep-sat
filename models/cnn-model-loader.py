import os
import time
import numpy as np
import scipy.io
import tensorflow as tf


DATA_PATH = '/home/nikatsanka/Workspace/tensor-env/deep-sat-datasets/sat-4-full.mat'
TEST_BATCH_SIZE = 128


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


def cnn_model_loader():
    # test_batch_size = 500
    data = DeepSatData()
    test_images = data.test.images.reshape(10, 10000, 28, 28, 4)
    test_label = data.test.labels.reshape(10, 10000, 4)
    # restore the training graph
    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
        # First let's load meta graph and restore weights
        loader = tf.train.import_meta_graph('./results-for-1e128bs-test/trained_models/saved_at_step--2560.meta')
        loader.restore(sess, tf.train.latest_checkpoint('./results-for-1e128bs-test/trained_models/'))

        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("x:0")
        Y = graph.get_tensor_by_name("y_:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        # Now, access the optimizer that you want to run.
        # prediction = graph.get_tensor_by_name("pred:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")

        # pred_out = sess.run(prediction, feed_dict={X: test_images})
        test_acc = np.mean([sess.run(accuracy, feed_dict={X: test_images[im], Y: test_label[im], keep_prob: 1.0})
                            for im in range(10)])
        print("Testing Accuracy:", test_acc)


if __name__ == "__main__":
    start_time = time.time()
    cnn_model_loader()
    # load_mat_data()
    time_stop = "\n--- %s seconds ---" % (time.time() - start_time)
    print(time_stop)
