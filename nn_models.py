# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import lasagne


def build_cnn_v1(pic_size, lambd, dropout_rate=0.0):
    """
    originally used to classify MNIST digits
    5x5 CONV-RELU-POOL, 5x5 CONV-RELU-POOL, FC, FC
    lambd - regularization rate
    
    Returns
    -------
    network, train_fn, val_fn
    """
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
            W=lasagne.init.HeUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5), pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 8192 units
    
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout_rate),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout_rate),
            num_units=36,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y_var)
    loss = loss.mean() + lambd * \
        lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=alpha_var, beta1=mu_var)
    
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
    
    return network, train_fn, val_fn
    
def build_cnn_florian(pic_size, n_of_classes, lambd=0.0, dropout_rate=0.0, complexity=4):
    """
    Florian Muellerklein’s VGG-like network, described here:
    http://florianmuellerklein.github.io/cnn_streetview/
    http://ankivil.com/kaggle-first-steps-with-julia-chars74k-first-place-using-convolutional-neural-networks/

    Parameters
    ----------
    lambd : float32
        regularization rate
    complexity : int
        multiplier of number of filters and FC units

    Returns
    -------
    network, train_fn, val_fn
    """
    
    # Declare Theano symbolic variables
    alpha_var = T.fscalar("alpha")
    mu_var = T.fscalar("mu")
    x_var = T.tensor4("x") # inut var
    y_var = T.lvector("y") # target var
    
    network = lasagne.layers.InputLayer(shape=(None, 1, pic_size, pic_size),
                                            input_var=x_var)
    # CONV-RELU
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32*complexity, filter_size=(3, 3), pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))
    # CONV-RELU
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32*complexity, filter_size=(3, 3), pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))
    # POOL
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # CONV-RELU
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64*complexity, filter_size=(3, 3), pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))
    # CONV-RELU
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64*complexity, filter_size=(3, 3), pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))
    # POOL
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # CONV-RELU
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128*complexity, filter_size=(3, 3), pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))
    # CONV-RELU
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128*complexity, filter_size=(3, 3), pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))
    # CONV-RELU
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128*complexity, filter_size=(3, 3), pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))
    # POOL
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # FC
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout_rate),
            num_units=1024*complexity,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))
    # FC
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout_rate),
            num_units=1024*complexity,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))
    
    # And, finally, output layer with dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout_rate),
            num_units=n_of_classes,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.HeNormal(gain=1.0))
    
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y_var)
    loss = loss.mean() + lambd * \
        lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    params = lasagne.layers.get_all_params(network, trainable=True)
#    updates = lasagne.updates.adamax(loss, params, learning_rate=alpha_var, beta1=mu_var)
    updates = lasagne.updates.adam(loss, params, learning_rate=alpha_var, beta1=mu_var)
    
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

    test_fn = theano.function([x_var], [test_pred, test_prediction])
    
    return network, train_fn, val_fn, test_fn
