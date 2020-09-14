from __future__ import absolute_import, print_function, division

import far_ho as far
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import experiment_manager as em
import os
import seaborn as sbn
import numpy as np
import markov


def get_reward(chain, trainset, n_class):
    reward = np.zeros(n_class)
    for i in range(n_class):
        ind = chain.index(i)

        train_next = em.Dataset(trainset.data[ind:ind+1], trainset.target[ind:ind+1])

        tr_supplier = train_next.create_supplier(x, y)
        val_supplier = validation.create_supplier(x, y)
        test_supplier = test.create_supplier(x, y)
        tf.global_variables_initializer().run()

        tr_accs, val_accs, test_accs = [], [], []

        run(T, inner_objective_feed_dicts=tr_supplier, outer_objective_feed_dicts=val_supplier)
        tr_accs.append(accuracy.eval(tr_supplier())), val_accs.append(accuracy.eval(val_supplier()))
        test_accs.append(accuracy.eval(test_supplier()))
        print('training accuracy', tr_accs[-1])
        print('validation accuracy', val_accs[-1])
        print('test accuracy', test_accs[-1])
        print('learning rate', lr.eval())
        print('norm of examples weight', tf.norm(weights).eval())
        # print(n)
        print('-' * 50)

        reward[i]=val_accs[-1]

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
        for i in list(set(range(n_class))-set(C)):
            alpha_k[i]=d[i]/b[i]

        highest_state = np.argmax(alpha_k)
        gIndex[highest_state] = alpha_k[highest_state]
        C.append(highest_state)

    return gIndex


if __name__ == "__main__":

    sbn.set_style('whitegrid')

    tf.reset_default_graph()
    ss = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='y')

    batch = 1
    size = 1400

    datasets = em.load.semeion()
    datasets = em.Datasets.from_list(datasets)
    train1 = datasets.train
    datasets = em.load.opt()
    datasets = em.Datasets.from_list(datasets)
    train2 = datasets.train
    train3 = datasets.validation
    train44 = datasets.test
    train4 = em.Dataset(train44.data[0:size], train44.target[0:size])
    train5 = em.Dataset(train44.data[size:], train44.target[size:])
    train=[train1, train2, train3, train4, train5]

    datasets2 = em.load.mnist(folder=os.path.join(os.getcwd(), 'MNIST_DATA/mnist'), partitions=(.02143,0.02,))
    datasets2 = em.Datasets.from_list(datasets2)
    validation = datasets2.validation
    test = datasets2.test

    with tf.variable_scope('model'):
        h1 = tcl.fully_connected(x, 300)
        out = tcl.fully_connected(h1, datasets.train.dim_target)
        print('Ground model weights (parameters)')
        [print(e) for e in tf.model_variables()]
    with tf.variable_scope('inital_weight_model'):
        h1_hyp = tcl.fully_connected(x, 300,
                                    variables_collections=far.HYPERPARAMETERS_COLLECTIONS,
                                    trainable=False)
        out_hyp = tcl.fully_connected(h1_hyp, datasets.train.dim_target,
                                    variables_collections=far.HYPERPARAMETERS_COLLECTIONS,
                                    trainable=False)
        print('Initial model weights (hyperparameters)')
        [print(e) for e in far.utils.hyperparameters()]
    #     far.utils.remove_from_collection(far.GraphKeys.MODEL_VARIABLES, *far.utils.hyperparameters())

    # get an hyperparameter for weighting the examples for the inner objective loss (training error)
    weights = far.get_hyperparameter('ex_weights', tf.zeros(batch))

    # build loss and accuracy
    # inner objective (training error), weighted mean of cross entropy errors (with sigmoid to be sure is > 0)
    with tf.name_scope('errors'):
        tr_loss = tf.reduce_mean(tf.sigmoid(weights)*tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
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
                        init_dynamics_dict={v: h for v, h in zip(tf.model_variables(), far.utils.hyperparameters()[:4])})

    print('Variables (or tensors) that will store the values of the hypergradients')
    print(*far.hypergradients(), sep='\n')


    # run hyperparameter optimization
    T = 80 # performs 80 iteraitons of gradient descent on the training error (rise this values for better performances)
    # get data suppliers (could also be stochastic for SGD)

    n_class = 10
    numSet = 5

    chain = []
    count = []
    for i in range(numSet):
        cchain, ccount = markov.get_chain(train[i].target, n_class)
        chain.append(cchain)
        count.append(ccount)

    p = []
    for i in range(numSet):
        pp = markov.tran_matrix(chain[i], count[i], n_class)
        p.append(pp)

    print("Getting reward...")

    reward = []
    for i in range(numSet):
        r = get_reward(chain[i], train[i], n_class)
        reward.append(r)

    gIndex = []
    for i in range(numSet):
        g = gittins_index(reward[i], p[i], n_class)
        gIndex.append(g)

    tf.global_variables_initializer().run()
    val_supplier = validation.create_supplier(x, y)
    test_supplier = test.create_supplier(x, y)

    print("Start Training...")
    print("@"*50)

    for i in range(0, size, batch):
        r = np.array([gIndex[0][chain[0][i]]])
        for k in range(1, numSet):
            ggg = np.reshape(gIndex[k][chain[k][i]], (1,))
            r = np.concatenate((r, ggg), axis=0)
        next = np.argmax(r)
        train_next = em.Dataset(train[next].data[i:i+batch], train[next].target[i:i+batch])
        tr_supplier = train_next.create_supplier(x, y)

        tr_accs, val_accs, test_accs = [], [], []

        run(T, inner_objective_feed_dicts=tr_supplier, outer_objective_feed_dicts=val_supplier)
        tr_accs.append(accuracy.eval(tr_supplier())), val_accs.append(accuracy.eval(val_supplier()))
        test_accs.append(accuracy.eval(test_supplier()))
        print(i)
        print('training accuracy', tr_accs[-1])
        print('validation accuracy', val_accs[-1])
        print('test accuracy', test_accs[-1])
        print('learning rate', lr.eval())
        print('norm of examples weight', tf.norm(weights).eval())
        # print(n)
        print('-' * 50)




