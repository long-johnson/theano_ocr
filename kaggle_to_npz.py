# -*- coding: utf-8 -*-

#
# Load kaggle image dataset, resize and save as npz file
#

import os
import glob
import numpy as np
import PIL

kaggle_folder = "data/kaggle/"
train_folder = kaggle_folder + "train/"
test_folder = kaggle_folder + "test/"
trainlabels_file = kaggle_folder + "trainLabels.csv"
picsize = 32
n_of_classes = 62
size = picsize * picsize
seed = 1
train_val_ratio = 0.75

def folder_to_ndarray(folder, isTrain=True):
    arr = []
    if isTrain:
        rng = range(1, 6284)
    else:
        rng = range(6284, 12504)
    for img_idx in rng:
        img = PIL.Image.open(folder + str(img_idx) + '.Bmp').convert('L')
        img = img.resize((picsize, picsize), PIL.Image.ANTIALIAS)
        img_data = np.reshape(np.array(img.getdata(), dtype=np.uint8), size)
        arr.append(img_data)
    return np.array(arr, dtype=np.uint8).reshape((len(arr), 1, picsize, picsize))

def label_to_int(label):
    #print(label)
    label = ord(label[2])
    #print(label)
    if ord('0') <= label and label <= ord('9'):
        return np.int8(label - ord('0'))
    if ord('a') <= label and label <= ord('z'):
        return np.int8(10 + label - ord('a'))
    if ord('A') <= label and label <= ord('Z'):
        return np.int8(36 + label - ord('A'))
    raise ValueError("label is out of range")

# Get training images
print("Forming X...")
X = folder_to_ndarray(train_folder, isTrain=True)

# Get test images
print("Forming Xtest...")
Xtest = folder_to_ndarray(test_folder, isTrain=False)

# Get training labels
print("Forming Y...")
labels = np.loadtxt(trainlabels_file, dtype=np.str, delimiter=',', skiprows=1, usecols=[1])
Y = np.array(list(map(label_to_int, labels)), dtype=np.uint8)

# splitting X dataset into Xtrain and Xval
print("splitting into train and val...")
# array that maps label to indexes samples that have this label
print("mapping")
labelToIndexes = [[] for i in range(n_of_classes)]
for i in range(len(Y)):
    label = Y[i]
    labelToIndexes[label].append(i)
# shuffle
print("shuffling...")
np.random.seed(seed)
for i in range(n_of_classes):
    np.random.shuffle(labelToIndexes[i])
# split
print("splitting...")
Xtrain = np.empty((0, 1, picsize, picsize), dtype=np.uint8)
Xval = np.empty((0, 1, picsize, picsize), dtype=np.uint8)
Ytrain = np.empty(0, dtype=np.uint8)
Yval = np.empty(0, dtype=np.uint8)
for i in range(n_of_classes):
    split_index = int(train_val_ratio * len(labelToIndexes[i]))
    train_indexes = labelToIndexes[i][:split_index]
    val_indexes = labelToIndexes[i][split_index:]
    # train
    Xtrain = np.append(Xtrain, X[train_indexes], axis=0)
    Ytrain = np.append(Ytrain, Y[train_indexes])
    # val
    Xval = np.append(Xval, X[val_indexes], axis=0)
    Yval = np.append(Yval, Y[val_indexes])

print("Final shuffle of Xtrain...")
np.random.seed(seed)
indices = np.arange(len(Xtrain))
np.random.shuffle(indices)    
Xtrain = Xtrain[indices]
Ytrain = Ytrain[indices]
    
# saving .npz files
print("saving .npz files...")
np.savez_compressed(kaggle_folder + "Xtrain.npz", Xtrain)
np.savez_compressed(kaggle_folder + "Xval.npz", Xval)
np.savez_compressed(kaggle_folder + "Xtest.npz", Xtest)
np.savez_compressed(kaggle_folder + "Ytrain.npz", Ytrain)
np.savez_compressed(kaggle_folder + "Yval.npz", Yval)

print("Done")