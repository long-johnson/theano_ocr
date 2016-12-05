# -*- coding: utf-8 -*-
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

def label_to_char(label: int):
    if label <= 9:
        return str(label)
    return chr(65 + label - 10)

out_dir = "out/"
class_results_dir = out_dir + "classify_res/"
img_dir = "data/to_classify"
filename_params = "CNNflorian-adam_complexity=1_n_epochs=173, batch_size=512, alpha=0.0008, lambd=0.01, mu=0.9, acc_train=98.9, acc_val=97.1,dropout=0.2, seed=1.npz"

pic_size = 32

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

network, _, _, test_fn = nn_models.build_cnn_florian(pic_size, complexity=1)
npz_file = np.load(out_dir + filename_params)
params = [npz_file["arr_{}".format(i)] for i in range(len(npz_file.files))]
lasagne.layers.set_all_param_values(network, params)

print("model was built")

if not os.path.exists(class_results_dir):
    os.makedirs(class_results_dir)

# form label strings
class_labels = []
for i in range(36):
    class_labels.append(label_to_char(i))

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
        plt.savefig(class_results_dir + "{}_actual={}.jpg".format(step * batch_size + i, label_to_char(pred_labels[i])))
        plt.clf()
        #img_original[step * batch_size + i].save(image_dir + "res/" + "{}_actual={}.jpg".format(step * batch_size + i, label_to_char(pred_labels[i])))
    print("step = {}".format(step))
    step += 1
