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
lambd = 0.0    # regularization coefficient
mu = np.float32(0.9)        # momentum rate
dropout = 0.0


#
# Theano graph construction
#

# Declare Theano symbolic variables
alpha_var = T.fscalar("alpha")
mu_var = T.fscalar("mu")
x_var = T.tensor4("x") # inut var
y_var = T.lvector("y") # target var

velocity_w = theano.shared(np.zeros((n_feats, K), dtype=np.float32), name="vel_w")

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
costs = []
accuracies_train = []
accuracies_val = []
time_experiment_start = time.time()
lookback = 5
lookback_epoch = 0
for epoch in range(n_epochs):
    time_start = time.time()
    # pass through training data
    cost = 0.0
    n_train_batches = int(np.ceil(len(Xtrain) / batch_size))
    step = 0
    acc_train = 0.0
    #for Xbatch, Ybatch in iterate_minibatches(Xtrain, Ytrain, batch_size, shuffle=False):
    for Xbatch, Ybatch in datagen.flow(Xtrain, Ytrain, batch_size, shuffle=False):
        if step >= n_train_batches:
            break
        Xbatch = np.float32(Xbatch)
        train_fn(Xbatch, Ybatch, alpha, mu)
        batch_loss, pred_batch = val_fn(Xbatch, Ybatch)
        acc_train = acc_train + np.count_nonzero(np.equal(pred_batch, Ybatch))
        cost = cost + batch_loss
        step = step + 1
    cost = cost / n_train_batches
    acc_train = acc_train / len(Ytrain) * 100.0
    # pass through validation data
    acc_val = 0.0
    for Xbatch, Ybatch in iterate_minibatches(Xval, Yval, batch_size, shuffle=False):
        acc_val = acc_val + np.count_nonzero(np.equal(val_fn(Xbatch, Ybatch)[1], Ybatch))
    acc_val = acc_val / len(Yval) * 100.0
    # decrease learning rate on some condition
    if lookback_epoch >= lookback\
    and min(costs[-lookback:]) < cost:
    #and max(accuracies_val[-lookback:]) > acc_val:
    #and min(costs[-lookback:]) < cost:
    #and all(np.array(accuracies_val[-lookback:]) > acc_val):
        alpha = 0.5 * alpha
        lookback = lookback * 2
        lookback_epoch = 0
        mu = np.float32(0.99)
        #temp = mu * 1.05
    #    mu = temp if temp < 0.91 else 0.99            
    costs.append(cost)
    accuracies_train.append(acc_train)
    accuracies_val.append(acc_val)
    print ("epoch = {}, cost = {:.4f}, acc_train={:.2f}%, acc_val = {:.2f}%, iter_time = {:.3f} s"\
           .format(epoch, cost, acc_train, acc_val, time.time() - time_start))
    print ("alpha={}, mu={}".format(alpha, mu))
    lookback_epoch = lookback_epoch + 1
print()
print ("time elapsed, min")
print ((time.time() - time_experiment_start) / 60)
print("final cost")
print(costs[-1])
import winsound
#winsound.Beep(500,2000)


#raise Exception('exit')

#
# validation
#
n_correct_train = 0
for Xbatch, Ybatch in iterate_minibatches(Xtrain, Ytrain, batch_size):
    n_correct_train = n_correct_train + np.count_nonzero(np.equal(val_fn(Xbatch, Ybatch)[1], Ybatch))
acc_train = n_correct_train / len(Ytrain) * 100.0
print("train prediction")
print(acc_train)
n_correct_val = 0
for Xbatch, Ybatch in iterate_minibatches(Xval, Yval, batch_size):
    n_correct_val = n_correct_val + np.count_nonzero(np.equal(val_fn(Xbatch, Ybatch)[1], Ybatch))
acc_val = n_correct_val / len(Yval) * 100.0
print("val prediction")
print(acc_val)

#
# Training plots
#
# cost plot
suptitle = "CNNv1_n_epochs={}, batch_size={}, alpha={}, lambd={}, mu={}, acc_train={:.1f}, acc_val={:.1f},"\
           "dropout={}"\
           .format(n_epochs, batch_size, alpha, lambd, mu, acc_train, acc_val, dropout)
plt.suptitle(suptitle)
plt.plot(costs[0:], label="cost")
plt.legend()
ax = plt.gca()
ax.grid(True)
plt.savefig(out_dir + suptitle + ".png")
plt.show()

# acc plot
plt.suptitle(suptitle)
plt.plot(accuracies_train, label="acc_train")
plt.plot(accuracies_val, label="acc_val")
plt.legend(loc='lower right')
ax = plt.gca()
ax.grid(True)
plt.savefig(out_dir + "accuracy " + suptitle + ".png")
plt.show()

#
# draw acc val, acc train plots
#
#print()
#print("started accumulating data for acc_val, acc_train plots")
#w_init = rng.randn(n_feats, K)
#b_init = np.zeros(K)
#xs = list(range(1, N, N // 10)) + [N]
#predicts_train = []
#predicts_val = []
#for i in xs:
#    w.set_value(w_init)
#    b.set_value(b_init)
#    for _ in range(training_steps):
#        train(Xtrain[:i], Ytrain[:i])
#    predicts_train += [100.0 - np.count_nonzero(np.equal(predict(Xtrain[:i]), Ytrain[:i])) / Ytrain[:i].size * 100.0]
#    predicts_val += [100.0 - np.count_nonzero(np.equal(predict(Xval), Yval)) / Yval.size * 100.0]
#    print ("Used {} train samples".format(i))
#
#plt.plot(xs, predicts_train)
#plt.plot(xs, predicts_val)
#plt.show()







# visualize first layer filters
params = lasagne.layers.get_all_param_values(network)
params_conv_l1 = params[0]
for i in range(32):
    plt.subplot(4, 8, i+1)
    params_conv_l1_visual = np.flipud(params_conv_l1[i, 0])
    plt.pcolor(params_conv_l1_visual, cmap=plt.cm.Greys_r)
    #plt.colorbar()
plt.show()







# save params
params = lasagne.layers.get_all_param_values(network)
np.savez(out_dir + suptitle + ".npz", *params)


#for i in range(len(params)):
#    np.save(out_dir + suptitle + "_" + str(i) + ".npy", params[i])

plt.pcolor(np.flipud(Xval[1000,0,:,:]), cmap=plt.cm.Greys_r)
plt.colorbar()
plt.show()

#for i in range(1):
#    rnd_idx = np.random.randint(0, len(Xval)-1)
#    idx = 200
#    xval = Xval[idx]
#    yval = Yval[idx]
#    print(predict([xval]))
#    print(yval)
#    plt.pcolor(np.flipud(xval.reshape((pic_size, pic_size))), cmap=plt.cm.gray)
#    plt.colorbar()
#    plt.show()

# save params
#np.save(out_dir + "W_"+suptitle+".npy", W_data)
#np.save(out_dir + "b_"+suptitle+".npy", b_data)

#
# Visualise weights
#
#for k in range(K):
#    W_visual = np.flipud(W_data[:, k].reshape((pic_size, pic_size)))
#    plt.pcolor(W_visual)
#    plt.colorbar()
#    plt.show()
    
    

