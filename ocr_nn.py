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

dataset_dir = "fnt_dataset/"
out_dir = "out/"
Xtrain_file = "XTrain_u.npz"
Xval_file = "XVal_u.npz"
Xtest_file = "XTest_u.npz"
Ytrain_file = "YTrain_u.npz"
Yval_file = "YVal_u.npz"
Ytest_file = "YTest_u.npz"

Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = None, None, None, None, None, None
gc.collect()
if Xtrain is None:
    Xtrain = np.array(np.load(dataset_dir + Xtrain_file)["arr_0"], dtype=np.float32)
if Ytrain is None:
    Ytrain = np.array(np.load(dataset_dir + Ytrain_file)["arr_0"], dtype=np.int8)
if Xval is None:
    Xval = np.array(np.load(dataset_dir + Xval_file)["arr_0"], dtype=np.float32)
if Yval is None:
    Yval = np.array(np.load(dataset_dir + Yval_file)["arr_0"], dtype=np.int8)
if Xtest is None:
    Xtest = np.array(np.load(dataset_dir + Xtest_file)["arr_0"], dtype=np.float32)
if Ytest is None:
    Ytest = np.array(np.load(dataset_dir + Ytest_file)["arr_0"], dtype=np.int8)

# shuffle train data
indices = np.arange(len(Xtrain))
np.random.shuffle(indices)    
Xtrain = Xtrain[indices]
Ytrain = Ytrain[indices]

N = Xtrain.shape[0]         # n of training samples
n_feats = Xtrain.shape[1]   # n of features per sample
K = np.max(Yval) + 1        # number of classes
pic_size = int(np.sqrt(n_feats))

# reshape data
Xtrain = Xtrain.reshape((len(Xtrain), 1, pic_size, pic_size))
Xval = Xval.reshape((len(Xval), 1, pic_size, pic_size))
Xtest = Xtest.reshape((len(Xtest), 1, pic_size, pic_size))
Ytrain = np.array(Ytrain, dtype=np.int64)
Yval = np.array(Yval, dtype=np.int64)
Ytest = np.array(Ytest, dtype=np.int64)

#
# learning params
#

seed = 1
n_epochs = 500
batch_size = 512
alpha_init = 8e-4 # learning rate
alpha = np.float32(alpha_init)     
lambd = 0.01    # regularization coefficient
mu_init = 0.9
mu = np.float32(mu_init)        # momentum rate
dropout_rate = 0.2


#
# Setting seed to reproduce results!
#
np.random.seed(seed)

#
# Theano graph construction
#
network, train_fn, val_fn, _ = nn_models.build_cnn_florian(pic_size, lambd,
                                                           dropout_rate=dropout_rate,
                                                           complexity=1)


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
        
# keras data augentation generator
datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.4,
    zoom_range=0.3,
    channel_shift_range=0.1,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

datagen.fit(Xtrain)

#
# Training
#
print("training start")
train_costs = []
train_accs = []
val_costs = []
val_accs = []

time_experiment_start = time.time()
lookback = 10
lookback_epoch = 0


for epoch in range(n_epochs):
    time_start = time.time()
    # pass through augmented training data with dropout
    n_train_batches = int(np.ceil(len(Xtrain) / batch_size))
    # datagen_train = datagen.flow(Xtrain, Ytrain, batch_size, shuffle=False)
    datagen_train = iterate_minibatches(Xtrain, Ytrain, batch_size, shuffle=False)
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
    and min(train_costs[-lookback:]) < train_cost:
        alpha = 0.5 * alpha
        # lookback += 5
        lookback_epoch = 0
        mu = np.float32(0.99)
    # save data to plot it later
    train_costs.append(train_cost)
    train_accs.append(train_acc)
    val_costs.append(val_cost)
    val_accs.append(val_acc)
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
print("final train cost")
print(train_costs[-1])
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

val_acc = 0
for Xbatch, Ybatch in iterate_minibatches(Xval, Yval, batch_size):
    val_acc += np.count_nonzero(np.equal(val_fn(Xbatch, Ybatch)[1], Ybatch))
val_acc = 100.0 * val_acc / len(Yval)
print("final val prediction")
print(val_acc)






#
# cost plot
#
plt.figure()
suptitle = "CNNflorian-adam_complexity=1_n_epochs={}, batch_size={}, alpha={}, lambd={}, mu={}, acc_train={:.1f}, acc_val={:.1f},"\
           "dropout={}, seed={}"\
           .format(len(train_costs), batch_size, alpha_init, lambd, mu_init,
                   train_acc, val_acc, dropout_rate, seed)
plt.suptitle(suptitle)
plt.plot(train_costs, label="train cost")
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


test_acc = 0
for Xbatch, Ybatch in iterate_minibatches(Xtest, Ytest, batch_size):
    test_acc += np.count_nonzero(np.equal(val_fn(Xbatch, Ybatch)[1], Ybatch))
test_acc = 100.0 * test_acc / len(Ytest)
print("final test prediction")
print(test_acc)
