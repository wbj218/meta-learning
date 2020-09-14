from __future__ import absolute_import, print_function, division
from functools import reduce
import experiment_manager as em
import numpy as np
import markov
import pickle


def get_data(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_dataset(dict, coarse_label, superclass):
    data_index = np.where(coarse_label==superclass)[0]
    data = dict[b'data'][data_index].reshape(len(data_index),32,32,3)
    # data = data / 255.
    label = np.array(dict[b'fine_labels'])[data_index]
    u = np.unique(label)
    target = np.zeros(shape=(len(data_index),num_class))
    for i in range(len(label)):
        ii = np.where(u==label[i])
        target[i,ii] = 1
    return em.Dataset(data, target)

if __name__=="__main__":
    numSet = 4
    num_class = 5
    numData = 2500
    SuperClass = [1, 7, 12, 16]

    file = 'cifar-100-python/train'
    train_dict = get_data(file)
    train_coarse_label = np.array(train_dict[b'coarse_labels'])
    train = []
    for i in range(numSet):
        t = get_dataset(train_dict,train_coarse_label,SuperClass[i])
        train.append(t)

    chain, count = [], []
    for i in range(numSet):
        ch, co = markov.get_chain(train[i].target, num_class)
        chain.append(ch)
        count.append(co)

    p = []
    for i in range(numSet):
        pp = markov.tran_matrix(chain[i], count[i], num_class)
        p.append(pp)

    f = []
    for i in range(numSet):
        ff = markov.count_matrix(chain[i], num_class)
        f.append(ff)

    for i in range(numSet):
        pv = markov.chi_square_test(f[i], p[i], count[i], num_class)
        print(pv)