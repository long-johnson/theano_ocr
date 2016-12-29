# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import sys
import os
import gc
import time
import os
import lasagne
import keras
from keras.preprocessing import image
import winsound

import nn_models

dataset_dir = "data/kaggle/"
out_dir = "out/"
Xtrain_file = "XTrain.npz"
Xval_file = "XVal.npz"
Xtest_file = "XTest.npz"
Ytrain_file = "YTrain.npz"
Yval_file = "YVal.npz"

Xtrain, Ytrain, Xval, Yval, Xtest = None, None, None, None, None
gc.collect()
Xtrain = np.array(np.load(dataset_dir + Xtrain_file)["arr_0"], dtype=np.float32)
Ytrain = np.array(np.load(dataset_dir + Ytrain_file)["arr_0"], dtype=np.int32)
Xval = np.array(np.load(dataset_dir + Xval_file)["arr_0"], dtype=np.float32)
Yval = np.array(np.load(dataset_dir + Yval_file)["arr_0"], dtype=np.int32)
# Xtest = np.array(np.load(dataset_dir + Xtest_file)["arr_0"], dtype=np.float32)

N = Xtrain.shape[0]         # n of training samples
K = np.max(Yval) + 1        # number of classes
pic_size = Xtrain.shape[2]

#
# learning params
#

unite_train_val = True  # whether we need to train on the united train/val set
seed = 1
n_epochs = 1000
batch_size = 128
alpha_init = 1e-4 # learning rate
alpha = np.float32(alpha_init)     
lambd = 0.02    # regularization coefficient
mu_init = 0.9
mu = np.float32(mu_init)        # momentum rate
dropout_rate = 0.5
complexity=4
lookback = 2    # initial number of previous costs to look at

if unite_train_val:
    Xtrain = np.append(Xtrain, Xval, axis=0)
    Ytrain = np.append(Ytrain, Yval, axis=0)
    Xval, YVal = None, None


# keras data augentation generator
datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=20, # tried 1, 2, 10, 20 succesfully
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.05,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    channel_flip=True,
    channel_flip_max=255.0)

# datagen.fit(Xtrain, seed=seed)


#
# Setting seed to reproduce results!
#
np.random.seed(seed)

#
# Theano graph construction
#
network, train_fn, val_fn, _ = nn_models.build_cnn_florian(pic_size,
                                                           n_of_classes=K,
                                                           lambd=lambd,
                                                           dropout_rate=dropout_rate,
                                                           complexity=complexity,
                                                           )

# shuffling is slow
def iterate_minibatches(X, Y, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
    for start_idx in range(0, len(X), batch_size):
        if shuffle:
            excerpt = indices[start_idx : start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield X[excerpt], Y[excerpt]


#
# Training
#
print("training start")
train_costs = []
train_accs = []
if not unite_train_val:
    val_costs = []
    val_accs = []

time_experiment_start = time.time()
lookback_epoch = 0


for epoch in range(n_epochs):
    time_start = time.time()
    # pass through augmented training data with dropout
    n_train_batches = int(np.ceil(len(Xtrain) / batch_size))
    datagen_train = datagen.flow(Xtrain, Ytrain, batch_size, shuffle=False, seed=(epoch))
    #datagen_train = iterate_minibatches(Xtrain, Ytrain, batch_size, shuffle=False)
    for step in range(n_train_batches):
        Xbatch, Ybatch = next(datagen_train)
        Xbatch = np.float32(Xbatch)
        train_fn(Xbatch, Ybatch, alpha, mu)
    # pass through unmodified training data without dropout
    train_acc = 0.0
    train_cost = 0.0
    for Xbatch, Ybatch in iterate_minibatches(Xtrain, Ytrain, batch_size, shuffle=False):
        cost, pred = val_fn(Xbatch, Ybatch)
        train_acc += np.count_nonzero(np.equal(pred, Ybatch))
        train_cost += cost
    train_acc = 100.0 * train_acc / len(Ytrain)
    train_cost /= n_train_batches
    # pass through validation data without dropout
    if not unite_train_val:
        val_acc = 0.0
        val_cost = 0.0
        n_val_batches = int(np.ceil(len(Xval) / batch_size))
        for Xbatch, Ybatch in iterate_minibatches(Xval, Yval, batch_size, shuffle=False):
            cost, pred = val_fn(Xbatch, Ybatch)
            val_acc += np.count_nonzero(np.equal(pred, Ybatch))
            val_cost += cost
        val_acc = 100.0 * val_acc / len(Yval)
        val_cost /= n_val_batches
    # decrease learning rate on some condition
    if lookback_epoch >= lookback\
    and max(train_costs[-lookback:]) < train_cost: # TODO: max ???
        alpha = 0.5 * alpha
        lookback += 1
        lookback_epoch = 0
        #if epoch > 100:
        #    mu = np.float32(0.99)
    # save data to plot it later
    train_costs.append(train_cost)
    train_accs.append(train_acc)
    if not unite_train_val:
        val_costs.append(val_cost)
        val_accs.append(val_acc)
    else:
        val_cost = 0.0
        val_acc = 0.0
    print ("epoch = {}, train_cost = {:.4f}, val_cost = {:.2f}, train_acc={:.2f}%, "\
           "val_acc = {:.2f}%, iter_time = {:.3f} s"\
           .format(epoch, train_cost, val_cost, train_acc, val_acc, time.time() - time_start))
    print ("alpha={}, mu={}".format(alpha, mu))
    lookback_epoch += 1
    if np.isnan(train_cost):
        winsound.Beep(500, 2000)
        raise OverflowError("train cost is nan")
print()
print ("time elapsed, min")
print ((time.time() - time_experiment_start) / 60)
winsound.Beep(500, 2000)



#
# validation
#
train_acc = 0
for Xbatch, Ybatch in iterate_minibatches(Xtrain, Ytrain, batch_size):
    train_acc += np.count_nonzero(np.equal(val_fn(Xbatch, Ybatch)[1], Ybatch))
train_acc = 100.0 * train_acc / len(Ytrain)
print("final train prediction")
print(train_acc)

val_acc = 0.0
if not unite_train_val:
    for Xbatch, Ybatch in iterate_minibatches(Xval, Yval, batch_size):
        val_acc += np.count_nonzero(np.equal(val_fn(Xbatch, Ybatch)[1], Ybatch))
    val_acc = 100.0 * val_acc / len(Yval)
    print("final val prediction")
    print(val_acc)






#
# cost plot
#
plt.figure()
suptitle = "testval_aug_kaggle_CNNflorian-adam_complexity={}_n_epochs={}, batch_size={}, alpha={}, lambd={}, mu={}, acc_train={:.1f}, acc_val={:.1f},"\
           "dropout={}, seed={}"\
           .format(complexity, len(train_costs), batch_size, alpha_init, lambd, mu_init,
                   train_acc, val_acc, dropout_rate, seed)
plt.suptitle(suptitle)
plt.plot(train_costs, label="train cost")
if not unite_train_val:
    plt.plot(val_costs, label="val cost")
plt.legend()
ax = plt.gca()
ax.grid(True)
plt.savefig(out_dir + suptitle + ".png")
plt.show()

#
# accuracies plot
#
plt.figure()
plt.suptitle(suptitle)
plt.plot(train_accs, label="train_accs")
if not unite_train_val:
    plt.plot(val_accs, label="val_accs")
plt.legend(loc='lower right')
ax = plt.gca()
ax.grid(True)
plt.savefig(out_dir + "accuracy " + suptitle + ".png")
plt.show()







#
# save params
#
params = lasagne.layers.get_all_param_values(network)
np.savez(out_dir + suptitle + ".npz", *params)







#
# visualize first layer filters
#
plt.figure()
params = lasagne.layers.get_all_param_values(network)
params_conv_l1 = params[0]
for i in range(32):
    plt.subplot(4, 8, i+1)
    params_conv_l1_visual = np.flipud(params_conv_l1[i, 0])
    plt.pcolor(params_conv_l1_visual, cmap=plt.cm.Greys_r)
    #plt.colorbar()
plt.show()

plt.figure()
params = lasagne.layers.get_all_param_values(network)
params_conv_l1 = params[4]
for i in range(64):
    plt.subplot(8, 8, i+1)
    params_conv_l1_visual = np.flipud(params_conv_l1[i, 0])
    plt.pcolor(params_conv_l1_visual, cmap=plt.cm.Greys_r)
    #plt.colorbar()
plt.show()

#
# visualize FC layer
#
#plt.figure()
#plt.pcolor(np.flipud(params[2]), cmap=plt.cm.Greys_r)
#plt.colorbar()
#plt.show()
