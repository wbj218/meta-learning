import numpy
import random
import tensorflow.contrib.learn.python.learn.datasets.mnist as mnist
import tensorflow.contrib.learn.python.learn.datasets.base as base
from tensorflow.python.framework import dtypes

def scale(im, nR, nC):
    nR0 = len(im)     # source number of rows
    nC0 = len(im[0])  # source number of columns
    return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
            for c in range(nC)] for r in range(nR)]

def read_semeion(fname = 'semeion/semeion.data'):
    file = open(fname, 'r')
    lines = file.readlines()

    width = 16
    height = 16
    size = width * height
    classes = 10

    images = []
    labels = []
    fnumber = 0

    for line in lines:
        data = line.split(' ')
        image = []
        label = []

        for i in range(0, size):
            image.append(int(float(data[i])))
        images.append(image)

        for i in range(size, size + classes):
            label.append(int(float(data[i])))
        labels.append(label)

        fnumber += 1

    for i in range(len(images)):
        ii = scale(numpy.reshape(images[i], (width, height)), 28, 28)
        ii = numpy.reshape(ii, (28, 28, 1))
        images[i] = ii

    width = 28
    height = 28

    # Shuffle data
    images_shuffle = []
    labels_shuffle = []
    indexes = list(range(len(images)))
    random.shuffle(indexes)
    for i in indexes:
        images_shuffle.append(images[i])
        labels_shuffle.append(labels[i])

    images = images_shuffle
    labels = labels_shuffle

    for i in range(len(labels)):
        labels[i] = numpy.reshape(labels[i], (10,))

    samples = len(lines)
    train_samples = 1400
    val_samples = 120
    test_samples = 73

    # Train set
    image_train = numpy.array(images[:train_samples], dtype=numpy.float32)
    image_train = image_train.reshape(train_samples, width, height, 1)

    label_train = numpy.array(labels[:train_samples], dtype=numpy.float32)

    # Validation Set
    image_val = numpy.array(images[train_samples:train_samples + val_samples], dtype=numpy.float32)
    image_val = image_val.reshape(val_samples, width, height, 1)

    label_val = numpy.array(labels[train_samples:train_samples + val_samples], dtype=numpy.float32)

    # test set
    image_test = numpy.array(images[train_samples + val_samples:], dtype=numpy.float32)
    image_test = image_test.reshape(test_samples, width, height, 1)

    label_test = numpy.array(labels[train_samples + val_samples:], dtype=numpy.float32)

    options = dict(dtype=dtypes.float32, reshape=True, seed=None)

    train = mnist.DataSet(image_train, label_train, **options)
    validation = mnist.DataSet(image_val, label_val, **options)
    test = mnist.DataSet(image_test, label_test, **options)

    return base.Datasets(train=train, validation=validation, test=test)


def read_opt(fname='optical/optdigits_csv.csv'):
    file = open(fname, 'r')
    lines = file.readlines()
    lines = lines[1:]
    width = 8
    height = 8
    size = width * height
    classes = 10

    images = []
    labels = []
    fnumber = 0

    for line in lines:
        data = line.split(',')
        image = []

        for i in range(0, size):
            image.append(int(float(data[i])))
        images.append(image)

        label = numpy.zeros((10,))
        label[int(data[-1])] = 1
        labels.append(label)

        fnumber += 1

    images_scale = [[]] * len(images)
    for i in range(len(images)):
        im_8 = numpy.reshape(images[i], (8, 8))
        im_reshape = scale(im_8, 28, 28)
        images_scale[i] = numpy.reshape(im_reshape, -1)

    images = images_scale

    # Shuffle data
    images_shuffle = []
    labels_shuffle = []
    indexes = list(range(len(images)))
    random.shuffle(indexes)
    for i in indexes:
        images_shuffle.append(images[i])
        labels_shuffle.append(labels[i])

    images = images_shuffle
    labels = labels_shuffle

    samples = len(images)

    width = 28
    height = 28

    train_samples = 1400
    val_samples = 1400
    test_samples = 2800

    # Train set
    image_train = numpy.array(images[:train_samples], dtype=numpy.float32)
    image_train = image_train.reshape(train_samples, width, height, 1)
    label_train = numpy.array(labels[:train_samples], dtype=numpy.float32)

    # Validation Set
    image_val = numpy.array(images[train_samples:train_samples + val_samples], dtype=numpy.float32)
    image_val = image_val.reshape(val_samples, width, height, 1)
    label_val = numpy.array(labels[train_samples:train_samples + val_samples], dtype=numpy.float32)

    # test set
    image_test = numpy.array(images[train_samples + val_samples: train_samples + val_samples + test_samples], dtype=numpy.float32)
    image_test = image_test.reshape(test_samples, width, height, 1)
    label_test = numpy.array(labels[train_samples + val_samples: train_samples + val_samples + test_samples], dtype=numpy.float32)

    options = dict(dtype=dtypes.float32, reshape=True, seed=None)

    train = mnist.DataSet(image_train, label_train, **options)
    validation = mnist.DataSet(image_val, label_val, **options)
    test = mnist.DataSet(image_test, label_test, **options)

    return base.Datasets(train=train, validation=validation, test=test)