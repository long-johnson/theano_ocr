# -*- coding: utf-8 -*-

""" Classify all the images in a folder and save predictions to another
folder. A prediction is a .jpg file named accordingly to the predicted class
label and jpg image contains the image of the sample and the barplot of
a prediction vector
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
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
img_dir = "data/kaggle/train"
filename_params = "aug1_kaggle_CNNflorian-adam_complexity=4_n_epochs=200, batch_size=128, alpha=0.0001, lambd=0.02, mu=0.9, acc_train=93.0, acc_val=79.1,dropout=0.5, seed=1.npz"

pic_size = 32
K = 62

img_original = []
X = []

for infile in glob.glob(os.path.join(img_dir, '*.*')):#+ ld[i]
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

# form label strings
class_labels = []
for i in range(K):
    class_labels.append(int_to_label(i))

# save pretty classification results
step = 0
for Xbatch in iterate_minibatches(X, batch_size):
    pred_labels, pred_vectors = test_fn(Xbatch)
    for i in range(len(Xbatch)):
        plt.subplot(1, 2, 1)
        plt.imshow(img_original[step * batch_size + i])
        plt.axis("off")
        plt.subplot(1, 2, 2)
        #fig, ax = plt.subplots()
        plt.barh(range(len(class_labels)), pred_vectors[i][::-1])
        plt.yticks(range(len(class_labels)), class_labels[::-1], rotation="vertical")
        plt.savefig(class_results_dir + "{}_actual={}.jpg".format(step * batch_size + i, int_to_label(pred_labels[i])))
        plt.clf()
        #img_original[step * batch_size + i].save(image_dir + "res/" + "{}_actual={}.jpg".format(step * batch_size + i, label_to_char(pred_labels[i])))
    print("step = {}".format(step))
    step += 1
