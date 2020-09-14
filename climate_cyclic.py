from __future__ import absolute_import, print_function, division

import far_ho as far
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import experiment_manager as em

from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

import h5py
import numpy as np


def scale(im, nR, nC):
    nR0 = len(im)  # source number of rows
    nC0 = len(im[0])  # source number of columns
    return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)] for c in range(nC)] for r in range(nR)]


def getOne(image, box):
    global numSet
    global scl
    global sample_set
    global label_set
    global feature  # 1, 3, 4, 7, 9, 11,

    for i in range(numSet):
        yrange = box[i, 0], box[i, 2] + 1
        xrange = box[i, 1], box[i, 3] + 1
        label = box[i, -1]
        label = np.reshape(label, [1, 1])
        ii = image[:, yrange[0]:yrange[1], xrange[0]:xrange[1]]
        sample = np.zeros([scl, scl, 1])
        for j in range(len(feature)):
            s = scale(ii[j], scl, scl)
            s = np.reshape(s, [scl, scl, 1])
            sample = np.concatenate((sample, s), axis=2)
        sample = sample[:, :, 1:]
        sample = np.reshape(sample, [1, scl, scl, len(feature)])
        sample_set[i] = np.concatenate((sample_set[i], sample), axis=0)
        label_set[i] = np.concatenate((label_set[i], label), axis=0)


def make_dataset(sample_set, label_set):
    sample_train = np.array(sample_set, dtype=np.float32)
    label_train = np.zeros((len(label_set), n_class))
    for i in range(len(label_set)):
        label_train[i][int(label_set[i])] = 1
    options = dict(dtype=dtypes.float32, reshape=True, seed=None)
    train = DataSet(sample_train, label_train, **options)
    validation = DataSet(sample_train, label_train, **options)
    test = DataSet(sample_train, label_train, **options)

    dataset = base.Datasets(train=train, validation=validation, test=test)

    train = em.Dataset(dataset.train.images, dataset.train.labels, name="CLIMATE")
    validation = em.Dataset(dataset.validation.images, dataset.validation.labels, name="CLIMATE")
    test = em.Dataset(dataset.test.images, dataset.test.labels, name="CLIMATE")
    res = [train, validation, test]
    return em.Datasets.from_list(res)


def get_reward(chain, trainset, n_class):
    reward = np.zeros(n_class)
    for i in range(n_class):
        if i in chain:
            ind = chain.index(i)

            train_next = em.Dataset(trainset.data[ind:ind + 1], trainset.target[ind:ind + 1])

            tr_supplier = train_next.create_supplier(x, y)
            val_supplier = val.create_supplier(x, y)
            # test_supplier = test.create_supplier(x, y)
            tf.global_variables_initializer().run()

            # tr_accs, val_accs, test_accs = [], [], []

            # hyper_step(T, inner_objective_feed_dicts=tr_supplier, outer_objective_feed_dicts=val_supplier)
            # res = sess.run(far.hyperparameters()) + [accuracy.eval(tr_supplier()), accuracy.eval(val_supplier())]

            tr_accs, val_accs = [], []

            run(T, inner_objective_feed_dicts=tr_supplier, outer_objective_feed_dicts=val_supplier)
            tr_accs.append(accuracy.eval(tr_supplier())), val_accs.append(accuracy.eval(val_supplier()))

            print('training accuracy', tr_accs[-1])
            print('validation accuracy', val_accs[-1])
            print('-' * 50)

            reward[i] = val_accs[-1]
        else:
            reward[i] = 0

    return reward


def gittins_index(reward, P, n_class):
    gIndex = np.zeros(n_class)
    C = []
    highest_state = np.argmax(reward)
    C.append(highest_state)
    gIndex[highest_state] = reward[highest_state]

    while len(C) < n_class:
        Q = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                if j in C:
                    Q[i][j] = P[i][j]

        d = np.dot(np.linalg.inv(np.identity(n_class) - np.multiply(0.99999, Q)), reward)
        b = np.dot(np.linalg.inv(np.identity(n_class) - np.multiply(0.99999, Q)), np.ones(n_class))

        alpha_k = np.zeros(n_class)
        for i in list(set(range(n_class)) - set(C)):
            alpha_k[i] = d[i] / b[i]

        highest_state = np.argmax(alpha_k)
        gIndex[highest_state] = alpha_k[highest_state]
        C.append(highest_state)

    return gIndex


if __name__=="__main__":
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    global feature
    # feature = [1,3,4,7,9,11,13]
    feature = [13]

    # get data from ExtremeWeather website
    # data_path = "/Volumes/TOSHIBA EXT/data/h5data/climo_2005.h5"
    h5f = h5py.File(data_path, 'r')
    images = h5f["images"][:, feature]
    boxes = h5f["boxes"]

    global numSet
    global scl
    global sample_set
    global label_set
    global numData
    global n_class

    numSet = 5
    scl = 32

    numData = 500
    n_class = 4

    batch = 1

    index = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        w = 15 - np.count_nonzero(box[:, 0] == -1)
        if w >= 5:
            index.append(i)

    print("finish indexing")
    sample_set = []
    label_set = []
    for i in range(numSet):
        sample_set.append(np.zeros((1, scl, scl, len(feature))))
        label_set.append(np.zeros((1, 1)))

    for i in range(50,numData+50):
        print(i)
        image = images[index[i]]
        box = boxes[index[i]]
        getOne(image, box)

    train = []
    for i in range(numSet):
        ex_set = sample_set[i][1:, :, :, :]
        la_set = label_set[i][1:, :]
        Dset = make_dataset(ex_set, la_set)
        train.append(Dset.train)

    print(train[0].data.shape)
    print(train[0].target.shape)

    val_set = np.zeros((1, scl, scl, len(feature)))
    val_labelset = np.zeros((1, 1))
    for i in range(50):
        val_im = images[index[i]]
        val_box = boxes[index[i]]
        w = 15 - np.count_nonzero(val_box[:, 0] == -1)
        for j in range(w):
            val_yrange = val_box[j, 0], val_box[j, 2] + 1
            val_xrange = val_box[j, 1], val_box[j, 3] + 1
            val_label = val_box[j, -1]
            val_label = np.reshape(val_label, [1, 1])
            val_ii = val_im[:, val_yrange[0]:val_yrange[1], val_xrange[0]:val_xrange[1]]
            val_sample = np.zeros([scl, scl, 1])
            for k in range(len(feature)):
                s = scale(val_ii[k], scl, scl)
                s = np.reshape(s, [scl, scl, 1])
                val_sample = np.concatenate((val_sample, s), axis=2)
            val_sample = val_sample[:, :, 1:]
            val_sample = np.reshape(val_sample, [1, scl, scl, len(feature)])
            val_set = np.concatenate((val_set, val_sample), axis=0)
            val_labelset = np.concatenate((val_labelset, val_label), axis=0)

    val = make_dataset(val_set[1:, :, :, :], val_labelset[1:, :])
    val = val.train
    print(val.data.shape)
    print(val.target.shape)

    test_set = np.zeros((1, scl, scl, len(feature)))
    test_labelset = np.zeros((1, 1))
    for i in range(numData + 50, numData + 200):
        test_im = images[index[i]]
        test_box = boxes[index[i]]
        w = 15 - np.count_nonzero(test_box[:, 0] == -1)
        for j in range(w):
            test_yrange = test_box[j, 0], test_box[j, 2] + 1
            test_xrange = test_box[j, 1], test_box[j, 3] + 1
            test_label = test_box[j, -1]
            test_label = np.reshape(test_label, [1, 1])
            test_ii = test_im[:, test_yrange[0]:test_yrange[1], test_xrange[0]:test_xrange[1]]
            test_sample = np.zeros([scl, scl, 1])
            for k in range(len(feature)):
                s = scale(test_ii[k], scl, scl)
                s = np.reshape(s, [scl, scl, 1])
                test_sample = np.concatenate((test_sample, s), axis=2)
            test_sample = test_sample[:, :, 1:]
            test_sample = np.reshape(test_sample, [1, scl, scl, len(feature)])
            test_set = np.concatenate((test_set, test_sample), axis=0)
            test_labelset = np.concatenate((test_labelset, test_label), axis=0)

    test = make_dataset(test_set[1:, :, :, :], test_labelset[1:, :])
    test = test.train
    print(test.data.shape)
    print(test.target.shape)

    x = tf.placeholder(tf.float32, shape=(None, scl * scl), name='x')
    y = tf.placeholder(tf.float32, shape=(None, n_class), name='y')

    with tf.variable_scope('model'):
        h1 = tcl.fully_connected(x, 300)
        out = tcl.fully_connected(h1, n_class)
        print('Ground model weights (parameters)')
        [print(e) for e in tf.model_variables()]
    with tf.variable_scope('inital_weight_model'):
        h1_hyp = tcl.fully_connected(x, 300,
                                     variables_collections=far.HYPERPARAMETERS_COLLECTIONS,
                                     trainable=False)
        out_hyp = tcl.fully_connected(h1_hyp, n_class,
                                      variables_collections=far.HYPERPARAMETERS_COLLECTIONS,
                                      trainable=False)
        print('Initial model weights (hyperparameters)')
        [print(e) for e in far.utils.hyperparameters()]

    weights = far.get_hyperparameter('ex_weights', tf.zeros(batch))

    with tf.name_scope('errors'):
        tr_loss = tf.reduce_mean(tf.sigmoid(weights) * tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
        # outer objective (validation error) (not weighted)
        val_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(out, 1)), tf.float32))

    # optimizers
    # get an hyperparameter for the learning rate
    lr = far.get_hyperparameter('lr', 0.01)
    io_optim = far.GradientDescentOptimizer(lr)  # for training error minimization an optimizer from far_ho is needed
    oo_optim = tf.train.AdamOptimizer()  # for outer objective optimizer all optimizers from tf are valid

    print('hyperparameters to optimize')
    [print(h) for h in far.hyperparameters()]

    # build hyperparameter optimizer
    farho = far.HyperOptimizer()
    run = farho.minimize(val_loss, oo_optim, tr_loss, io_optim,
                         init_dynamics_dict={v: h for v, h in
                                             zip(tf.model_variables(), far.utils.hyperparameters()[:4])})

    print('Variables (or tensors) that will store the values of the hypergradients')
    print(*far.hypergradients(), sep='\n')

    T = 100

    next = 0
    val_supplier = val.create_supplier(x, y)
    test_supplier = test.create_supplier(x, y)
    tf.global_variables_initializer().run()

    for i in range(0, numData, batch):
        if next == numSet:
            next = 0
        train_next = em.Dataset(train[next].data[i:i + batch], train[next].target[i:i + batch])
        next += 1
        tr_supplier = train_next.create_supplier(x, y)
        tr_accs, val_accs, test_accs = [], [], []

        run(T, inner_objective_feed_dicts=tr_supplier, outer_objective_feed_dicts=val_supplier)
        tr_accs.append(accuracy.eval(tr_supplier())), val_accs.append(accuracy.eval(val_supplier()))
        test_accs.append(accuracy.eval(test_supplier()))

        print(next)
        print('training accuracy', tr_accs[-1])
        print('validation accuracy', val_accs[-1])
        print('test accuracy', test_accs[-1])
        print('-' * 50)

