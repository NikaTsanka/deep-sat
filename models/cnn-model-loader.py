import time
import numpy as np
import scipy.io
import tensorflow as tf
import os
import random
np.set_printoptions(suppress=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


DATA_PATH = '/home/nikatsanka/Workspace/tensor-env/deep-sat-datasets/sat-6-full.mat'
# IMAGE_PATH = '/home/nikatsanka/Dropbox/school-work/machine-learning/term-proj/deep-sat-ml-proj/models/single-test-image-sat4.mat'
IMAGE_PATH_SAT4 = '/home/nikatsanka/Dropbox/school-work/machine-learning/term-proj/deep-sat-ml-proj/models/100-test-image-sat4.mat'
IMAGE_PATH_SAT6 = '/home/nikatsanka/Dropbox/school-work/machine-learning/term-proj/deep-sat-ml-proj/models/100-test-image-sat6.mat'


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


def cnn_model_predict_sat4():
    import matplotlib.pyplot as plt
    labels = ('barren land', 'trees', 'grassland', 'none')
    # test_batch_size = 500
    test_data = scipy.io.loadmat(IMAGE_PATH_SAT4)
    test_images = test_data['image']
    test_labels = test_data['label']

    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
        # First let's load meta graph and restore weights
        loader = tf.train.import_meta_graph('./results-for-1e128bs-test1/trained_models/model.ckpt-2560.meta')
        loader.restore(sess, tf.train.latest_checkpoint('./results-for-1e128bs-test1/trained_models/'))

        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("x:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        # Now, access the optimizer that you want to run.
        prediction = graph.get_tensor_by_name("pred:0")
        # accuracy = graph.get_tensor_by_name("accuracy:0")

        pred_out = []

        random_indexes = []

        for i in range(5):
            random_indexes.append(random.randint(0, 100))

        for i in random_indexes:
            # print(random.randint(0, 101))
            print(i)
            pred_out.append(sess.run(prediction, feed_dict={X: test_images[i][np.newaxis, ...], keep_prob: 1.0})[0])

        width = 0.5

        def autolabel(rects, ax):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                         '%f' % float(height), ha='center', va='bottom')

        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

        for im, row in enumerate(ax):
            print(im)
            for i, col in enumerate(row):
                # print(i)
                if i == 0:
                    print(random_indexes[im])
                    col.imshow(test_images[random_indexes[im]])  # [..., :3]
                    col.set_ylabel(labels[int(np.argmax(test_labels[random_indexes[im]]))])
                else:
                    l = col.bar(labels, pred_out[im], width, color='#f49141')
                    col.set_ylabel('Prediction')
                    autolabel(l, col)
        plt.show()


def cnn_model_predict_sat6():
    import matplotlib.pyplot as plt
    labels = ('building', 'barren land', 'trees', 'grassland', 'road', 'water')
    # test_batch_size = 500
    test_data = scipy.io.loadmat(IMAGE_PATH_SAT6)
    test_images = test_data['image']
    test_labels = test_data['label']

    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
        # First let's load meta graph and restore weights
        loader = tf.train.import_meta_graph('/home/nikatsanka/Dropbox/school-work/machine-learning/term-proj/j-notebook/results-for-7e128bs-JamilaNet-sat6/trained_models/model.ckpt-17280.meta')
        loader.restore(sess, tf.train.latest_checkpoint('/home/nikatsanka/Dropbox/school-work/machine-learning/term-proj/j-notebook/results-for-7e128bs-JamilaNet-sat6/trained_models/'))

        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("x:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        phase_train = graph.get_tensor_by_name("phase_train:0")

        # Now, access the optimizer that you want to run.
        prediction = graph.get_tensor_by_name("pred:0")
        # accuracy = graph.get_tensor_by_name("accuracy:0")

        pred_out = []

        random_indexes = []

        for i in range(5):
            random_indexes.append(random.randint(0, 100))

        for i in random_indexes:
            # print(random.randint(0, 101))
            # print(i)
            pred_out.append(sess.run(prediction, feed_dict={X: test_images[i][np.newaxis, ...], keep_prob: 1.0, phase_train: False})[0])

        width = 0.5

        def autolabel(rects, ax):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                         '%f' % float(height), ha='center', va='bottom')

        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

        for im, row in enumerate(ax):
            # print(im)
            for i, col in enumerate(row):
                # print(i)
                if i == 0:
                    # print(random_indexes[im])
                    col.imshow(test_images[random_indexes[im]])  # [..., :3]
                    col.set_ylabel(labels[int(np.argmax(test_labels[random_indexes[im]]))])
                else:
                    l = col.bar(labels, pred_out[im], width, color='#f49141')
                    col.set_ylabel('Prediction')
                    autolabel(l, col)
        plt.show()


def cnn_model_predict_acc():
    # test_batch_size = 500
    data = DeepSatData()
    test_images = data.test.images.reshape(10, 10000, 28, 28, 4)
    test_label = data.test.labels.reshape(10, 10000, 4)
    # restore the training graph
    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
        # First let's load meta graph and restore weights
        loader = tf.train.import_meta_graph('./results-for-1e128bs-test2/trained_models/saved_at_step--2560.meta')
        loader.restore(sess, tf.train.latest_checkpoint('./results-for-1e128bs-test2/trained_models/'))

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


def test_data_writer():
    data = DeepSatData()
    '''
    (100000, 28, 28, 4)
    (100000, 4)
    (10, 10000, 28, 28, 4)
    (10, 10000, 4)
    (1, 28, 28, 4)
    (1, 4)
    '''
    # print(data.test.images.shape)
    # print(data.test.labels.shape)
    # test_images = data.test.images.reshape(10, 10000, 28, 28, 4)
    # test_label = data.test.labels.reshape(10, 10000, 4)
    # print(test_images.shape)
    # print(test_label.shape)

    batch_im = data.test.next_batch(100)
    # print(batch_im.shape)
    print(batch_im[0].shape)
    print(batch_im[1].shape)
    # print(batch_im[1])
    # _im = batch_im[0][-1:]
    # _lb = batch_im[1][-1:]
    _im = batch_im[0][:]
    _lb = batch_im[1][:]
    print(_im.shape)
    print(_lb.shape)

    im = {'image': _im, 'label': _lb}
    scipy.io.savemat('100-test-image-sat6', im)

    '''
    (100, 28, 28, 4)
    (100, 4)
    (100, 28, 28, 4)
    (100, 4)
    '''


if __name__ == "__main__":
    start_time = time.time()
    # cnn_model_loader()
    # test_data_writer()
    # cnn_model_predict_sat4()
    cnn_model_predict_sat6()
    time_stop = "\n--- %s seconds ---" % (time.time() - start_time)
    print(time_stop)
