from __future__ import absolute_import, print_function, division

import far_ho as far
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import experiment_manager as em
import os
import seaborn as sbn

sbn.set_style('whitegrid')

tf.reset_default_graph()
ss = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')

batch = 100
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

#i=0

acc=[]

train=[train1, train2, train3, train4, train5]

val_supplier = validation.create_supplier(x, y)
test_supplier = test.create_supplier(x, y)
tf.global_variables_initializer().run()

index = 0

for i in range(0, size, batch):
    if index == 5:
        index = 0
    train_next = em.Dataset(train[index].data[i:i+batch], train[index].target[i:i+batch])
    tr_supplier = train_next.create_supplier(x, y)
    index += 1
    tr_accs, val_accs, test_accs = [], [], []

    run(T, inner_objective_feed_dicts=tr_supplier, outer_objective_feed_dicts=val_supplier)
    tr_accs.append(accuracy.eval(tr_supplier())), val_accs.append(accuracy.eval(val_supplier()))
    test_accs.append(accuracy.eval(test_supplier()))
    print(index, i)
    print('training accuracy', tr_accs[-1])
    print('validation accuracy', val_accs[-1])
    print('test accuracy', test_accs[-1])
    print('learning rate', lr.eval())
    print('norm of examples weight', tf.norm(weights).eval())
    # print(n)
    print('-' * 50)

    acc.append(test_accs[-1])

    #print(acc)
    print('*' * 50)