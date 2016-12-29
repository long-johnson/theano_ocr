# -*- coding: utf-8 -*-

""" Classify all the images in a folder and save predictions to another
folder. A prediction is a .jpg file named accordingly to the predicted class
label and jpg image contains the image of the sample and the barplot of
a prediction vector
"""

import os
import numpy as np
from PIL import Image
import lasagne
import nn_models


def iterate_minibatches(X, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
    for start_idx in range(0, len(X), batch_size):
        if shuffle:
            excerpt = indices[start_idx : start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield X[excerpt]


def int_to_label(label_int):
    if label_int < 10:
        return str(label_int)
    if 10 <= label_int < 36:
        return chr(ord('a') + label_int - 10)
    if 36 <= label_int < 62:
        return chr(ord('A') + label_int - 36)
    raise ValueError("Label index is out of range")

out_dir = "out/"
class_results_dir = out_dir + "classify_res/"
img_dir = "data/kaggle/test/"
filename_params = "testval_aug_kaggle_CNNflorian-adam_complexity=4_n_epochs=1105, batch_size=128, alpha=0.0001, lambd=0.02, mu=0.9, acc_train=96.8, acc_val=0.0,dropout=0.5, seed=1.npz"

pic_size = 32
K = 62
min_index = 6284
max_index = 12504

img_original = []
X = []

for i in range(min_index, max_index):
    infile = img_dir + "{}.Bmp".format(i)
    img_original.append(Image.open(infile).convert('RGB'))
    im = Image.open(infile).convert('L')
    im = im.resize((pic_size, pic_size), Image.ANTIALIAS)
    im = np.reshape(np.array(im.getdata(), dtype=np.float32), pic_size*pic_size)
    X.append(im)

X = np.array(X)
X = X.reshape((len(X), 1, pic_size, pic_size))
batch_size = 128

print("start")

network, _, _, test_fn = nn_models.build_cnn_florian(pic_size, n_of_classes=K, complexity=4)
npz_file = np.load(out_dir + filename_params)
params = [npz_file["arr_{}".format(i)] for i in range(len(npz_file.files))]
lasagne.layers.set_all_param_values(network, params)

print("model was built")

if not os.path.exists(class_results_dir):
    os.makedirs(class_results_dir)

# save pretty classification results
all_pred_labels = []
step = 0
for Xbatch in iterate_minibatches(X, batch_size):
    pred_labels, _ = test_fn(Xbatch)
    for pred in pred_labels:
        all_pred_labels.append(int_to_label(pred))
    print("step = {}".format(step))
    step += 1

indexes = list(range(min_index, max_index))

import csv
with open(out_dir + "kaggle_predictions.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['ID', 'Class'])
    for i in range(max_index-min_index):
        csv_writer.writerow([indexes[i], all_pred_labels[i]])
    

#np.savetxt(out_dir + "kaggle_predictions.csv",
#           np.asarray(indexes, np.array(all_pred_labels, dtype=np.string_), dtype=np.string_).T,
#           delimiter=',', header='ID,Class')
