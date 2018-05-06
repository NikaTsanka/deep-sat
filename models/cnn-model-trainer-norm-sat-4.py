import os
import time
import numpy as np
import scipy.io
import tensorflow as tf
from models.layers import conv_layer, max_pool_2x2, full_layer, conv_layer_no_relu, avg_pool_2x2


# DATA_PATH = '/home/nikatsanka/Workspace/tensor-env/deep-sat-datasets/sat-4-full.mat'
DATA_PATH = ''.join([os.getcwd(), '/', 'sat-4-full.mat'])
# DATA_PATH = 'dataset/sat-4-full.mat'

# HYPERS
NUM_SAMPLES = 400000
EPOCHS = 7
BATCH_SIZE = 128
STEPS = int((NUM_SAMPLES * EPOCHS) / BATCH_SIZE)
ONE_EPOCH = int(NUM_SAMPLES / BATCH_SIZE)
TEST_INTERVAL = BATCH_SIZE * 5
MODELS_TO_KEEP = 5
lr = 0.0001
decay = 0.9
momentum = 0
dropoutProb = 0.5

LABELS = os.path.join(os.getcwd(), "metadata-sat4.tsv")  # Label path for visualization
SPRITES = os.path.join(os.getcwd(), "sprite-sat4.png")

version = 'JamilaNet-sat6'
output_dir = 'results-for-' + str(EPOCHS) + 'e' + str(BATCH_SIZE) + 'bs-' + version
log_dir = os.path.join(output_dir, 'logs')
log_name = 'lr' + str(lr) + 'd' + str(decay) + 'm' + str(momentum) + 'do' + str(dropoutProb)
output_file = 'output.txt'
model_dir = os.path.join(output_dir, 'trained_models')
model_name = 'model.ckpt'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

write_to_file = open(os.path.join(output_dir, output_file), 'w')
write_to_file.write('HYPER_PARAMETERS_USED')
write_to_file.write('\n---------------------')
write_to_file.write('\nNUM_SAMPLES:' + str(NUM_SAMPLES))
write_to_file.write('\nEPOCHS:' + str(EPOCHS))
write_to_file.write('\nBATCH_SIZE:' + str(BATCH_SIZE))
write_to_file.write('\nSTEPS:' + str(STEPS))
write_to_file.write('\nLEARNING_RATE:' + str(lr))
write_to_file.write('\nDECAY:' + str(decay))
write_to_file.write('\nMOMENTUM:' + str(momentum))
write_to_file.write('\nDROPOUT:' + str(dropoutProb))


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7

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

    def random_batch(self, batch_size):
        n = len(self.images)
        ix = np.random.choice(n, batch_size)
        return self.images[ix], self.labels[ix]


class DeepSatData:
    def __init__(self):
        self.train = DeepSatLoader('train').load_data()
        self.test  = DeepSatLoader('test').load_data()


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """

    beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train, mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def cnn_model_trainer():
    # ALEXNET
    dataset = DeepSatData()

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 4], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 4], name='y_')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    conv1 = conv_layer_no_relu(x, shape=[3, 3, 4, 16], pad='SAME')
    conv1_bn = batch_norm(conv1, 16, phase_train)
    conv1_rl = tf.nn.relu(conv1_bn)
    # conv1_pool = max_pool_2x2(conv1_rl, 2, 2) #28x28x4->14x14x16

    conv2 = conv_layer_no_relu(conv1_rl, shape=[3, 3, 16, 32], pad='SAME')
    conv2_bn = batch_norm(conv2, 32, phase_train)
    conv2_rl = tf.nn.relu(conv2_bn)
    conv2_pool = avg_pool_2x2(conv2_rl, 2, 2)  # 14x14x16->7x7x32

    conv3 = conv_layer(conv2_pool, shape=[3, 3, 32, 64], pad='SAME')
    conv3_bn = batch_norm(conv3, 64, phase_train)
    conv3_rl = tf.nn.relu(conv3_bn)
    conv3_pool = avg_pool_2x2(conv3_rl, 2, 2)  # 7x7x32 ->7x7x64

    conv4 = conv_layer(conv3_pool, shape=[3, 3, 64, 96], pad='SAME')
    conv4_bn = batch_norm(conv4, 96, phase_train)
    conv4_rl = tf.nn.relu(conv4_bn)
    # conv4_pool = max_pool_2x2(conv4) # 7x7x64 -> 7x7x96

    conv5 = conv_layer(conv4_rl, shape=[3, 3, 96, 64], pad='SAME')
    conv5_bn = batch_norm(conv5, 64, phase_train)
    conv5_rl = tf.nn.relu(conv5_bn)
    conv5_pool = avg_pool_2x2(conv5_rl, 2, 2)  # 7x7x96 ->7x7x64

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

    # pred = tf.nn.softmax(logits=full_3, name='pred')  # for later prediction
    pred_out = tf.argmax(full_3, 1, name='pred')

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=full_3, labels=y_))

    # train_step = tf.train.RMSPropOptimizer(lr, decay, momentum).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(pred_out, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    # Setting up for the visualization of the data in Tensorboard
    embedding_size = 200    # size of second to last fc layer
    embedding_input = full_2    # FC2 as input
    # Variable containing the points in visualization
    embedding = tf.Variable(tf.zeros([10000, embedding_size]), name="test_embedding")
    assignment = embedding.assign(embedding_input)  # Will be passed in the test session

    merged_sum = tf.summary.merge_all()

    def test(test_sess, assign):
        x_ = dataset.test.images.reshape(10, 10000, 28, 28, 4)
        y = dataset.test.labels.reshape(10, 10000, 4)

        test_acc = np.mean([test_sess.run(accuracy, feed_dict={x: x_[im], y_: y[im], keep_prob: 1.0, phase_train: False})
                            for im in range(10)])

        # Pass through the last 10,000 of the test set for visualization
        test_sess.run([assign], feed_dict={x: x_[9], y_: y[9], keep_prob: 1.0, phase_train: False})
        return test_acc

    # config=config
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tensorboard
        sum_writer = tf.summary.FileWriter(os.path.join(log_dir, log_name))
        sum_writer.add_graph(sess.graph)

        # Create a Saver object
        # max_to_keep: keep how many models to keep. Delete old ones.
        saver = tf.train.Saver(max_to_keep=MODELS_TO_KEEP)

        # setting up Projector
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name
        embedding_config.metadata_path = LABELS     # labels

        # Specify the width and height of a single thumbnail.
        embedding_config.sprite.image_path = SPRITES
        embedding_config.sprite.single_image_dim.extend([28, 28])
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(sum_writer, config)

        for i in range(STEPS):
            batch = dataset.train.random_batch(BATCH_SIZE)
            # batch = dataset.train.next_batch(BATCH_SIZE)
            batch_x = batch[0]
            batch_y = batch[1]

            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropoutProb, phase_train: True})

            _, summ = sess.run([train_step, merged_sum], feed_dict={x: batch_x, y_: batch_y, keep_prob: dropoutProb, phase_train: True})
            sum_writer.add_summary(summ, i)

            if i % ONE_EPOCH == 0:
                ep_print = "\n*****************EPOCH: %d" % ((i/ONE_EPOCH) + 1)
                write_to_file.write(ep_print)
                print(ep_print)
            if i % TEST_INTERVAL == 0:
                acc = test(sess, assignment)
                loss = sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, phase_train: False})
                ep_test_print = "\nEPOCH:%d" % ((i/ONE_EPOCH) + 1) + " Step:" + str(i) + \
                                "|| Minibatch Loss= " + "{:.4f}".format(loss) + \
                                " Accuracy: {:.4}%".format(acc * 100)
                write_to_file.write(ep_test_print)
                print(ep_test_print)
                # Create a checkpoint in every iteration
                saver.save(sess, os.path.join(model_dir, model_name),
                           global_step=i)

        test(sess, assignment)
        sum_writer.close()


def create_tsv(ds):
    # Creates the labels for the last 10,000 pictures
    labels = ['barren land', 'trees', 'grassland', 'none']
    with open('metadata.tsv', 'w') as f:
        y = ds.test.labels[-10000:]
        for i in range(10000):
            argmax = int(np.argmax(y[i]))
            f.write("%s \n" % labels[argmax])


def create_sprite_image(images):
    # print(type(images))
    # if isinstance(images, list):
    #     print('TRUE')
    #     images = np.array(images)
    img_h = images.shape[1]  # 28
    img_w = images.shape[2]  # 28
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))  # 100

    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots, 3))  # (2800, 2800)

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def mat_data_load_test():
    # import matplotlib.pyplot as plt
    #
    # data = DeepSatData()

    # (10000, 28, 28, 3)
    # ps = data.test.images[-10000:][..., :3]
    #
    # sprite = create_sprite_image(ps)
    # plt.imsave('sprite-sat4.png', sprite, cmap='Greys')
    #
    # print(ps.shape)
    # print(data.train.images.shape)
    # print(data.train.labels.shape)
    # (400000, 28, 28, 4)
    # (400000, 4)


    # create_tsv(data)

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
    dataset = scipy.io.loadmat(DATA_PATH)
    # train_x = dataset['train_x']
    # train_y = dataset['train_y']
    #
    # test_x = dataset['test_x']
    # test_y = dataset['test_y']

    labels = dataset['annotations']

    # print(type(train_x))
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)

    print(labels)

    print(labels.shape)

    # for i in dataset.values():
    #     print(len(i))

    # print(type(dataset['train_y']))


if __name__ == "__main__":
    start_time = time.time()
    cnn_model_trainer()
    # mat_data_load_test()
    time_stop = "\n--- %s seconds ---" % (time.time() - start_time)
    write_to_file.write(time_stop)
    print(time_stop)
    write_to_file.close()
