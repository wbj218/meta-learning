from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.stats as stats


def get_chain(target, n_class):
    size = len(target)
    train_state = []
    for i in range(size):
        train_state.append(np.argmax(target[i]))

    state_count = []
    for i in range(n_class):
        state_count.append(train_state.count(i))

    return train_state, state_count


def tran_matrix(train_state, state_count, n_class):
    size = len(train_state)
    p = np.zeros((n_class, n_class))
    for m in range(size-1):
        i = train_state[m]
        j = train_state[m+1]
        p[i][j] += 1

    for i in range(n_class):
        if i in train_state:
            p[i] = [x/state_count[i] for x in p[i]]

    return p


def count_matrix(train_state,  n_class):
    size = len(train_state)

    f = np.zeros((n_class, n_class))
    for m in range(size - 1):

        i = train_state[m]
        j = train_state[m + 1]
        f[i][j] += 1

    return f


def chi_square_test(f, p, count, n_class):

    pv = np.zeros(n_class)
    for m in range(n_class):
        v = 0
        for i in range(n_class):
            for j in range(n_class):
                v = v + np.square(f[i][j] - count[i]*p[m][j])/(count[i]*p[m][j])

        df = np.square((n_class-1))
        pv[m] = stats.chi2.pdf(v, df)

    return max(pv)


def combine(chain_x, chain_y):
    size = len(chain_x)

    new_chain = np.zeros(size)
    for i in range(size):
        new_chain[i] = 10*chain_y[i]+chain_x[i]

    return new_chain


def check_independence(tar_x, tar_y, n_class):
    chain_x, count_x = get_chain(tar_x, n_class)
    chain_y, count_y = get_chain(tar_y, n_class)

    new_chain = combine(chain_x, chain_y)
    count = np.zeros(n_class*n_class)

    for l in range(n_class*n_class):
        count[l] = new_chain.count(l)

    f = count_matrix(new_chain, n_class*n_class)
    p_x = tran_matrix(chain_x, count_x, n_class)
    p_y = tran_matrix(chain_y, count_y, n_class)
    P = np.kron(p_y, p_x)

    v = 0
    for i in range(n_class*n_class):
        for j in range(n_class*n_class):
            v = v + np.square(f[i][j] - count[i]*P[i][j])/(count[i]*P[i][j])

    df = np.square((n_class*n_class-1))
    p = stats.chi2.pdf(v, df)
    return p





