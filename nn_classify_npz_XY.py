# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import lasagne
import nn_models

""" Load samples X and correct labels Y from .npz file, then
classify all the images in a .npz file and save all classified images to
another folder in .jpg files where file name displays correct and predicted
class labels. Also displays a percent of correctly classified images.
"""




out_dir = "out/"
#incorrect_dir = out_dir + "incorrect_dir/"
dataset_dir = "data/kaggle/"
filename_params = "aug1_kaggle_CNNflorian-adam_complexity=4_n_epochs=200, batch_size=128, alpha=0.0001, lambd=0.02, mu=0.9, acc_train=93.0, acc_val=79.1,dropout=0.5, seed=1.npz"
filename_X = "XVal.npz"
filename_Y = "YVal.npz"
incorrect_dir = out_dir + "incorrect_" + filename_X + "_" + filename_params[:-4] + "/"

batch_size = 128
pic_size = 32



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

def int_to_label(label_int):
    if label_int < 10:
        return str(label_int)
    if 10 <= label_int < 36:
        return chr(ord('a') + label_int - 10)
    if 36 <= label_int < 62:
        return chr(ord('A') + label_int - 36)
    raise ValueError("Label index is out of range")


X = np.array(np.load(dataset_dir + filename_X)["arr_0"], dtype=np.float32)
Y = np.array(np.load(dataset_dir + filename_Y)["arr_0"], dtype=np.int32)

print("start")

network, _, val_fn, _ = nn_models.build_cnn_florian(pic_size, n_of_classes=62, complexity=4)
npz_file = np.load(out_dir + filename_params)
params = [npz_file["arr_{}".format(i)] for i in range(len(npz_file.files))]
#params = 
lasagne.layers.set_all_param_values(network, params)

print("model was built")

if not os.path.exists(incorrect_dir):
    os.makedirs(incorrect_dir)
acc = 0
step = 0
for Xbatch, Ybatch in iterate_minibatches(X, Y, batch_size):
    _, pred = val_fn(Xbatch, Ybatch)
    correct_pred = np.equal(pred, Ybatch)
    for i in np.where(np.logical_not(correct_pred))[0]:
        result = Image.fromarray(Xbatch[i, 0, :, :].astype(np.uint8))
        result.save(incorrect_dir + "{}_correct={}_actual={}.jpg".format(step * batch_size + i, int_to_label(Ybatch[i]), int_to_label(pred[i])))
    acc += np.count_nonzero(correct_pred)
    print("step = {}".format(step))
    step += 1
acc = 100.0 * acc / len(Y)
print("final accuracy")
print(acc)
