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

dataset_dir = "data/font/"
out_dir = "out/"
Xtrain_file = "XTrain_u.npz"
Xval_file = "XVal_u.npz"
Ytrain_file = "YTrain_u.npz"
Yval_file = "YVal_u.npz"

Xtrain, Ytrain, Xval, Yval = None, None, None, None
gc.collect()
if Xtrain is None:
    Xtrain = np.array(np.load(dataset_dir + Xtrain_file)["arr_0"], dtype=np.float32)
if Ytrain is None:
    Ytrain = np.array(np.load(dataset_dir + Ytrain_file)["arr_0"], dtype=np.int8)
if Xval is None:
    Xval = np.array(np.load(dataset_dir + Xval_file)["arr_0"], dtype=np.float32)
if Yval is None:
    Yval = np.array(np.load(dataset_dir + Yval_file)["arr_0"], dtype=np.int8)

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
Ytrain = np.array(Ytrain, dtype=np.int64)
Yval = np.array(Yval, dtype=np.int64)

#
# learning params
#
n_epochs = 500
batch_size = 2048
alpha = np.float32(1e-4)     # learning rate
lambd = 1.0    # regularization coefficient
mu = np.float32(0.9)        # momentum rate
dropout = 0.5


#
# Theano graph construction
#

# Declare Theano symbolic variables
alpha_var = T.fscalar("alpha")
mu_var = T.fscalar("mu")
x_var = T.tensor4("x") # inut var
y_var = T.lvector("y") # target var

network = lasagne.layers.InputLayer(shape=(None, 1, pic_size, pic_size),
                                        input_var=x_var)
network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5), pad = 'same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5), pad = 'same',
        nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
# 8192 units

# A fully-connected layer of 256 units with 50% dropout on its inputs:
network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=dropout),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

# And, finally, the 10-unit output layer with 50% dropout on its inputs:
network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=dropout),
        num_units=36,
        nonlinearity=lasagne.nonlinearities.softmax)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_var)
loss = loss.mean() + lambd * \
       lasagne.regularization.regularize_network_params(network,
                                                        lasagne.regularization.l2)

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=alpha_var, momentum=mu_var)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        y_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_pred = T.argmax(test_prediction, axis=1)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([x_var, y_var, alpha_var, mu_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([x_var, y_var], [test_loss, test_pred])


# shuffling is slow
def iterate_minibatches(X, Y, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
    for start_idx in range(0, len(X) - batch_size + 1, batch_size):
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
train_costs = []
train_accs = []
val_costs = []
val_accs = []
time_experiment_start = time.time()
lookback = 5
lookback_epoch = 0
for epoch in range(n_epochs):
    time_start = time.time()
    # pass through augmented training data with dropout
    n_train_batches = int(np.ceil(len(Xtrain) / batch_size))
    datagen_train = datagen.flow(Xtrain, Ytrain, batch_size, shuffle=False)
    for step in range(n_train_batches):
        Xbatch, Ybatch = datagen_train.next()
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
    for Xbatch, Ybatch in iterate_minibatches(Xval, Yval, batch_size, shuffle=False):
        cost, pred = val_fn(Xbatch, Ybatch)
        val_acc += np.count_nonzero(np.equal(pred, Ybatch))
        val_cost += cost
    val_acc = 100.0 * val_acc / len(Yval)
    val_cost /= n_train_batches
    # decrease learning rate on some condition
    if lookback_epoch >= lookback\
    and min(train_costs[-lookback:]) < train_cost:
        alpha = 0.5 * alpha
        lookback = lookback * 2
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
print()
print ("time elapsed, min")
print ((time.time() - time_experiment_start) / 60)
print("final train cost")
print(train_costs[-1])
import winsound
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
suptitle = "CNNv1_n_epochs={}, batch_size={}, alpha={}, lambd={}, mu={}, acc_train={:.1f}, acc_val={:.1f},"\
           "dropout={}"\
           .format(n_epochs, batch_size, alpha, lambd, mu, train_acc, val_acc, dropout)
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

#
# visualize FC layer
#
plt.figure()
plt.pcolor(np.flipud(params[2]), cmap=plt.cm.Greys_r)
plt.colorbar()
plt.show()




