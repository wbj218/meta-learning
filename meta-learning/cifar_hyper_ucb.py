from __future__ import absolute_import, print_function, division
from functools import reduce
import far_ho as far
import experiment_manager as em
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import far_ho.examples as far_ex
from far_ho.examples.datasets import Dataset, Datasets
import os
from os.path import join
import numpy as np
import statistics as st
import random
import matplotlib.pyplot as plt
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import seaborn as sbn
from collections import defaultdict
import markov
import pickle
from PIL import Image

def get_data(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def hyper_conv_layer(x):
    hyper_coll = far.HYPERPARAMETERS_COLLECTIONS
    return tcl.conv2d(x, num_outputs=64, stride=2,
                      kernel_size=3,
                      # normalizer_fn=
                      # lambda z: tcl.batch_norm(z,
                      #   variables_collections=hyper_coll,
                      #   trainable=False),
                      trainable=False,
                      variables_collections=hyper_coll)

def build_hyper_representation(_x, auto_reuse=False):
    reuse = tf.AUTO_REUSE if auto_reuse else False
    with tf.variable_scope('HR', reuse=reuse):
        conv_out = reduce(lambda lp, k: hyper_conv_layer(lp),
                          range(4), _x)
        return tf.reshape(conv_out, shape=(-1, 256))

def classifier(_x, _y):
    return tcl.fully_connected(
        _x, int(_y.shape[1]), activation_fn=None,
        weights_initializer=tf.zeros_initializer)

def get_placeholders():
    _x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    _y = tf.placeholder(tf.float32, (None, num_class))
    return _x, _y

def make_feed_dicts(tasks, mbd):
    train_fd, test_fd = {}, {}
    for task, _x, _y in zip(tasks, mbd['x'], mbd['y']):
        train_fd[_x] = task.train.data
        train_fd[_y] = task.train.target
        test_fd[_x] = task.test.data
        test_fd[_y] = task.test.target
    return train_fd, test_fd

def accuracy(y_true, logits):
    return tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y_true, 1), tf.argmax(logits, 1)),
            tf.float32))

def meta_test(meta_batches, mbd, opt, n_steps):
    #ss = tf.get_default_session()
    #ss.run(tf.variables_initializer(tf.trainable_variables()))
    sum_loss, sum_acc = 0., 0.
    n_tasks = len(mbd['err'])*len(meta_batches)
    for _tasks in meta_batches:
        _train_fd, _valid_fd = make_feed_dicts(_tasks, mbd)
        mb_err = tf.add_n(mbd['err'])
        mb_acc = tf.add_n(mbd['acc'])
        opt_step = opt.minimize(mb_err)
        for i in range(n_steps):
            sess.run(opt_step, feed_dict=_train_fd)

        mb_loss, mb_acc = sess.run([mb_err, mb_acc], feed_dict=_valid_fd)
        sum_loss += mb_loss
        sum_acc += mb_acc

    return sum_loss/n_tasks, sum_acc/n_tasks

def get_dataset(dict, coarse_label, superclass):
    data_index = np.where(coarse_label==superclass)[0]
    data = dict[b'data'][data_index].reshape(len(data_index),32,32,3)
    label = np.array(dict[b'fine_labels'])[data_index]
    u = np.unique(label)
    target = np.zeros(shape=(len(data_index),num_class))
    for i in range(len(label)):
        ii = np.where(u==label[i])
        target[i,ii] = 1
    return em.Dataset(data, target)

if __name__=="__main__":
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    batch = 50
    numSet = 4
    num_class = 5
    numData = 2500
    SuperClass = [0, 7, 12, 16]
    meta_batch_size = 1
    lr = 0.01

    file = 'cifar-100-python/train'
    train_dict = get_data(file)
    train_coarse_label = np.array(train_dict[b'coarse_labels'])
    train = []
    val_data = np.zeros(shape=(1,32,32,3))
    val_target = np.zeros(shape=(1,num_class))
    for i in range(numSet):
        t = get_dataset(train_dict,train_coarse_label,SuperClass[i])
        val_index = np.random.choice(numData, 10, replace=False)
        val_data = np.concatenate([val_data,t.data[val_index]])
        val_target = np.concatenate([val_target, t.target[val_index]])
        if i==0:
            val_data = val_data[1:]
            val_target = val_target[1:]
        train.append(t)
        print(t.data.shape)
        print(t.target.shape)
    validation = em.Dataset(val_data, val_target)
    print(validation.data.shape)
    print(validation.target.shape)

    file = 'cifar-100-python/test'
    test_dict = get_data(file)
    test_coarse_label = np.array(test_dict[b'coarse_labels'])
    test_train = get_dataset(train_dict, train_coarse_label, 8)
    test_test = get_dataset(test_dict, test_coarse_label, 8)

    mb_dict = defaultdict(list)  # meta_batch dictionary
    for _ in range(meta_batch_size):
        x, y = get_placeholders()
        mb_dict['x'].append(x)
        mb_dict['y'].append(y)
        hyper_repr = build_hyper_representation(x, auto_reuse=True)
        logits = classifier(hyper_repr, y)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=logits))
        mb_dict['err'].append(ce)
        mb_dict['acc'].append(accuracy(y, logits))

    L = tf.add_n(mb_dict['err'])
    E = L / meta_batch_size
    mean_acc = tf.add_n(mb_dict['acc'])/meta_batch_size

    inner_opt = far.GradientDescentOptimizer(learning_rate=lr)
    outer_opt = tf.train.AdamOptimizer()

    hyper_step = far.HyperOptimizer().minimize(
        E, outer_opt, L, inner_opt)

    T = 10

    count = [1]*numSet
    acc = []

    for i in range(numSet):
        tf.global_variables_initializer().run()
        next = i
        train_next = em.Dataset(train[i].data[0:batch], train[i].target[0:batch])

        dd = em.Datasets.from_list([train_next, validation])
        train_fd, valid_fd = make_feed_dicts([dd]*meta_batch_size, mb_dict)
        hyper_step(T, train_fd, valid_fd)

        test_optim = tf.train.GradientDescentOptimizer(lr)
        test = em.Datasets.from_list([test_train, test_test])
        #test_mbs = [mb for mb in all_data.test.generate(n_episodes_testing, batch_size=meta_batch_size, rand=0)]

        EE, accu = sess.run([E, mean_acc], feed_dict=valid_fd)
        test_accu = meta_test([[test]*meta_batch_size], mb_dict, test_optim, T)

        print('train_test (loss, acc)', [EE, accu])
        print('test_test (loss, acc)', test_accu)
            
        print('-' * 50)

        acc.append(accu)

    t = numSet
    factor = numSet

    tf.global_variables_initializer().run()

    test_acc=[]
    val_acc = []

    for ii in range(10):
        for i in range(0, numData, batch*meta_batch_size):
            meta_batch = []
            for m in range(meta_batch_size):
                t+=1
                temp = 0
                c = 0
                for k in range(numSet):
                    e = acc[k] + np.sqrt((factor * np.log(t)) / (count[k]))
                    if e > temp:
                        temp = e
                        c = k
                count[c] += 1
                next = c
                train_next_data = train[next].data[i+m*batch:i+m*batch+batch]
                train_next_target = train[next].target[i+m*batch:i+m*batch+batch]

                #r = np.array([gIndex[0][chain[0][i]], gIndex[1][chain[1][i]], gIndex[2][chain[2][i]], gIndex[3][chain[3][i]], gIndex[4][chain[4][i]]])

                # ll = chain[next][i]
                # percentage[next][ll] += 1
                # train_next = far_ex.Dataset(train[next].data[i:i + 1], train[next].target[i:i + 1])
                train_next_batch = em.Dataset(train_next_data, train_next_target)
                dd = em.Datasets.from_list([train_next_batch, validation])
                meta_batch.append(dd)

            train_fd, valid_fd = make_feed_dicts(meta_batch, mb_dict)
            hyper_step(T, train_fd, valid_fd)

            test_optim = tf.train.GradientDescentOptimizer(lr)
            test = em.Datasets.from_list([test_train, test_test])

            EE, accu = sess.run([E, mean_acc], feed_dict=valid_fd)
            test_accu = meta_test([[test]*meta_batch_size], mb_dict, test_optim, T)

            print("Hyper Step: {}".format(i+ii*numData))
            print('train_test (loss, acc)', [EE, accu])
            print('test_test (loss, acc)', test_accu)
            print('-' * 50)

            acc[c] = (acc[c] * (count[c] - 1) + accu) / count[c]

            val_acc.append(accu)
            test_acc.append(test_accu[1])

    # with open('cifar_ucb_3.csv', 'w') as ofile:
    #     for i in range(len(val_acc)):
    #         line = str(val_acc[i]) + ',' + str(test_acc[i]) + '\n'
    #         # new_line = line.rstrip('\n') + ',' + str(new) + '\n'
    #         ofile.write(line)
