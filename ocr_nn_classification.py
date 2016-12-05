# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import lasagne
import nn_models


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

def label_to_char(label: int):
    if label <= 9:
        return str(label)
    return chr(65 + label - 10)

out_dir = "out/"
#incorrect_dir = out_dir + "incorrect_dir/"
dataset_dir = "data/font/"
filename_params = "CNNflorian-adam_complexity=1_n_epochs=173, batch_size=512, alpha=0.0008, lambd=0.01, mu=0.9, acc_train=98.9, acc_val=97.1,dropout=0.2, seed=1.npz"
filename_X = "XVal_u.npz"
filename_Y = "YVal_u.npz"
incorrect_dir = out_dir + "incorrect_" + filename_X + "_" + filename_params[:-4] + "/"


batch_size = 1024

X = np.array(np.load(dataset_dir + filename_X)["arr_0"], dtype=np.float32)
Y = np.array(np.load(dataset_dir + filename_Y)["arr_0"], dtype=np.int64)
n_feats = X.shape[1]   # n of features per sample
pic_size = int(np.sqrt(n_feats))
X = X.reshape((len(X), 1, pic_size, pic_size))

print("start")

network, _, val_fn, _ = nn_models.build_cnn_florian(pic_size, complexity=1)
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
        result.save(incorrect_dir + "{}_correct={}_actual={}.jpg".format(step * batch_size + i, label_to_char(Ybatch[i]), label_to_char(pred[i])))
    acc += np.count_nonzero(correct_pred)
    print("step = {}".format(step))
    step += 1
train_acc = 100.0 * acc / len(Y)
print("final train prediction")
print(train_acc)
