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
#incorrect_dir = out_dir + "incorrect_dir/"
image_dir = "D:/python_projects/image_dir/"
filename_params = "CNNflorian_complexity=1_n_epochs=65, batch_size=128, alpha=6.25000029685907e-05, lambd=0.01, mu=0.9900000095367432, acc_train=98.8, acc_val=88.3,dropout=0.05, seed=2.npz"

pic_size = 32

img_original = []
X = []

for infile in glob.glob(os.path.join(image_dir, '*.*')):#+ ld[i]
        img_original.append(Image.open(infile))
        im = Image.open(infile).convert('L')
        im = im.resize((pic_size, pic_size), Image.ANTIALIAS)
        #im.save(tmp_path, "PNG")
        
        #im = io.imread(tmp_path)
        #im = color.rgb2gray(im)
        #misc.imsave(path.format(i, j), im)
        im = np.reshape(np.array(im.getdata(), dtype=np.float32), pic_size*pic_size)
        X.append(im)

X = np.array(X)
X = X.reshape((len(X), 1, pic_size, pic_size))
batch_size = 128

print("start")

network, _, _, test_fn = nn_models.build_cnn_florian(pic_size, complexity=3)
npz_file = np.load(out_dir + filename_params)
params = [npz_file["arr_{}".format(i)] for i in range(len(npz_file.files))]
lasagne.layers.set_all_param_values(network, params)

print("model was built")

if not os.path.exists(image_dir + "res/"):
    os.makedirs(image_dir + "res/")

class_labels = []
for i in range(36):
    class_labels.append(label_to_char(i))

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
        plt.savefig(image_dir + "res/" + "{}_actual={}.jpg".format(step * batch_size + i, label_to_char(pred_labels[i])))
        plt.clf()
        #img_original[step * batch_size + i].save(image_dir + "res/" + "{}_actual={}.jpg".format(step * batch_size + i, label_to_char(pred_labels[i])))
    print("step = {}".format(step))
    step += 1
