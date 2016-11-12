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


dataset_dir = "fnt_dataset/"
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

#
# learning params
#
n_epochs = 1000
batch_size = 2000
alpha = 1e-4     # learning rate
lambd = 1000.0    # regularization coefficient
mu = 0.5        # momentum rate

#
# Theano graph construction
#
# Declare Theano symbolic variables
x = T.fmatrix("x")
y = T.bvector("y")
alpha_var = T.fscalar("alpha")
mu_var = T.fscalar("mu")

# initialize the weight vector w randomly
w = theano.shared(np.array(np.random.randn(n_feats, K), dtype=np.float32), name="w")

# initialize the bias term
b = theano.shared(np.zeros(K, dtype=np.float32), name="b")

# velocity for momentum
velocity_w = theano.shared(np.zeros((n_feats, K), dtype=np.float32), name="vel_w")
velocity_b = theano.shared(np.zeros(K, dtype=np.float32), name="vel_b")

# Construct Theano expression graph
softmax = T.nnet.softmax(T.dot(x, w) + b)
prediction = T.argmax(softmax, axis=1)                   
xent = -T.mean(T.log(softmax)[T.arange(y.shape[0]), y]) # softmax loss
cost = xent + lambd * (w ** 2).sum()    # The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)
#
# Compile
#
train = theano.function(
          inputs=[x, y, alpha_var, mu_var],
          outputs=[prediction, cost],
          updates=((velocity_w, mu_var*velocity_w - alpha_var*gw),
                   (velocity_b, mu_var*velocity_b - alpha_var*gb),
                   (w, w + velocity_w), (b, b + velocity_b),
                   )
          )
predict = theano.function(inputs=[x], outputs=prediction)

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

#
# Training
#
costs = []
accuracies_train = []
accuracies_val = []
time_experiment_start = time.time()
lookback_epoch = 0
for epoch in range(n_epochs):
    time_start = time.time()
    # pass through training data
    cost = 0.0
    n_train_batches = 0
    acc_train = 0.0
    for Xbatch, Ybatch in iterate_minibatches(Xtrain, Ytrain, batch_size, shuffle=False):
        pred_batch, batch_cost = train(Xbatch, Ybatch, alpha, mu)
        acc_train = acc_train + np.count_nonzero(np.equal(pred_batch, Ybatch))
        cost = cost + batch_cost
        n_train_batches = n_train_batches + 1
    cost = cost / n_train_batches
    acc_train = acc_train / len(Ytrain) * 100.0
    # pass through validation data
    acc_val = 0.0
    for Xbatch, Ybatch in iterate_minibatches(Xval, Yval, batch_size, shuffle=False):
        acc_val = acc_val + np.count_nonzero(np.equal(predict(Xbatch), Ybatch))
    acc_val = acc_val / len(Yval) * 100.0
    # decrease learning rate on some condition
    lookback = 10
    if lookback_epoch >= lookback\
    and min(costs[-lookback:]) < cost:
    #and max(accuracies_val[-lookback:]) > acc_val:
    #and min(costs[-lookback:]) < cost:
    #and all(np.array(accuracies_val[-lookback:]) > acc_val):
        alpha = 0.5 * alpha
        lookback = lookback * 2
        lookback_epoch = 0
        #temp = mu * 1.05
    #    mu = temp if temp < 0.91 else 0.99            
    costs.append(cost)
    accuracies_train.append(acc_train)
    accuracies_val.append(acc_val)
    print ("epoch = {}, cost = {:.2f}, acc_train={:.2f}%, acc_val = {:.2f}%, iter_time = {} s"\
           .format(epoch, cost, acc_train, acc_val, time.time() - time_start))
    print ("alpha={}, mu={}".format(alpha, mu))
    lookback_epoch = lookback_epoch + 1
print()
print ("time elapsed, min")
print ((time.time() - time_experiment_start) / 60)
print("final cost")
print(costs[-1])
import winsound
winsound.Beep(500,2000)


raise Exception('exit')

#
# validation
#
n_correct_train = 0
for Xbatch, Ybatch in iterate_minibatches(Xtrain, Ytrain, batch_size):
    n_correct_train = n_correct_train + np.count_nonzero(np.equal(predict(Xbatch), Ybatch))
acc_train = n_correct_train / len(Ytrain) * 100.0
print("train prediction")
print(acc_train)
n_correct_val = 0
for Xbatch, Ybatch in iterate_minibatches(Xval, Yval, batch_size):
    n_correct_val = n_correct_val + np.count_nonzero(np.equal(predict(Xbatch), Ybatch))
acc_val = n_correct_val / len(Yval) * 100.0
print("val prediction")
print(acc_val)

#
# Training plots
#
# cost plot
suptitle = "n_epochs={}, batch_size={}, alpha={}, lambd={}, mu={}, acc_train={:.1f}, acc_val={:.1f}"\
           .format(n_epochs, batch_size, alpha, lambd, mu, acc_train, acc_val)
plt.suptitle(suptitle)
plt.plot(costs[0:], label="cost")
plt.legend()
ax = plt.gca()
ax.grid(True)
plt.savefig(suptitle+".png")
plt.show()

# acc plot
plt.suptitle(suptitle)
plt.plot(accuracies_train, label="acc_train")
plt.plot(accuracies_val, label="acc_val")
plt.legend(loc='lower right')
ax = plt.gca()
ax.grid(True)
plt.savefig("accuracy "+suptitle+".png")
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

#
# Save weights
#
W_data = w.get_value()
b_data = b.get_value()
np.save("W_"+suptitle+".npy", W_data)
np.save("b_"+suptitle+".npy", b_data)

W_visual = np.flipud(W_data[:, 0].reshape((pic_size, pic_size)))
plt.pcolor(W_visual)
plt.colorbar()
plt.show()

#
# Visualise weights
#
for k in range(K):
    W_visual = np.flipud(W_data[:, k].reshape((pic_size, pic_size)))
    plt.pcolor(W_visual)
    plt.colorbar()
    plt.show()
    
    

