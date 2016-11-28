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
dataset_dir = "fnt_dataset/"
filename_params = "CNNv1_n_epochs=500, batch_size=512, alpha=1.000000013351432e-10, lambd=0.0, mu=0.8999999761581421, acc_train=0.0, acc_val=0.0,dropout=0.0, seed=2.npz"
incorrect_dir = out_dir + filename_params[:-4] + "/"
filename_X = "XTrain_u.npz"
filename_Y = "YTrain_u.npz"

batch_size = 1024

X = np.array(np.load(dataset_dir + filename_X)["arr_0"], dtype=np.float32)
Y = np.array(np.load(dataset_dir + filename_Y)["arr_0"], dtype=np.int64)
n_feats = X.shape[1]   # n of features per sample
pic_size = int(np.sqrt(n_feats))
X = X.reshape((len(X), 1, pic_size, pic_size))

print("start")

network, _, val_fn = nn_models.build_cnn_florian(pic_size, complexity=1)
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