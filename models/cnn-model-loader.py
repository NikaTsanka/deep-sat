import os
import time
import numpy as np
import scipy.io
import tensorflow as tf
from models.layers import conv_layer, max_pool_2x2, full_layer


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
    test_images = data.test.images[:TEST_BATCH_SIZE]
    test_images = np.dot((test_images[:, :, :, :])[..., :3], [0.299, 0.587, 0.114])
    test_label = data.test.labels[:TEST_BATCH_SIZE]
    # restore the training graph
    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
        # First let's load meta graph and restore weights
        loader = tf.train.import_meta_graph('./saved-models/model_iter-19600.meta')
        loader.restore(sess, tf.train.latest_checkpoint('./saved-models/'))

        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")

        # Now, access the optimizer that you want to run.
        # prediction = graph.get_tensor_by_name("pred:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")

        # pred_out = sess.run(prediction, feed_dict={X: test_images})
        test_acc = sess.run(accuracy, feed_dict={X: test_images, Y: test_label})
        print("Testing Accuracy:", test_acc)

if __name__ == "__main__":
    start_time = time.time()
    cnn_model_loader()
    # load_mat_data()
    time_stop = "\n--- %s seconds ---" % (time.time() - start_time)
    print(time_stop)
