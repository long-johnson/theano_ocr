# -*- coding: utf-8 -*-

import os, sys
import PIL
import random
import glob
from skimage import color
from skimage import io
import numpy as np
from scipy import misc

dirr = "data/font/"
dirr2 = dirr + "pics/"

ld = os.listdir(dirr2)
#ld = [1, 2]
size_x = 32
size_y = 32

size = size_x*size_y
s = size_x, size_y

#X = []
x_tmp_divided = []
x_tmp_united = []

y_tmp_divided = []
y_tmp_united = []

XTrain_d = []
XVal_d = []
XTest_d = []
YTrain_d = []
YVal_d = []
YTest_d = []

XTrain_u = []
XVal_u = []
XTest_u = []
YTrain_u = []
YVal_u = []
YTest_u = []

border_idx_divided = []
border_idx_united = []
idx = 0
for i in range(len(ld)):
    print(i)
    for infile in glob.glob(os.path.join(dirr2 + ld[i], '*.*')):#+ ld[i]
        im = PIL.Image.open(infile).convert('L')
        im = im.resize(s, PIL.Image.ANTIALIAS)
        #im.save(tmp_path, "PNG")
        
        #im = io.imread(tmp_path)
        #im = color.rgb2gray(im)
        #misc.imsave(path.format(i, j), im)
        im = np.reshape(np.array(im.getdata(), dtype=np.uint8), size)
        x_tmp_united.append(im)
        x_tmp_divided.append(im)
        
        #y_divided_vector = np.zeros(62, dtype=int)
        #y_united_vector = np.zeros(36, dtype=int)
        
        #y_divided_vector[i] = 1
        y_divided = np.uint8(i)
        y_tmp_divided.append(y_divided)

        y_united = np.uint8(i if i < 10 else 10 + idx)    
        y_tmp_united.append(y_united)
    if i >= 10 and i % 2 != 0:   
        idx = idx + 1
    #border_idx_divided.append(idx)
    if i < 10:
        #border_idx_united.append(idx)
        random.shuffle(x_tmp_united)
        random.shuffle(x_tmp_divided)
        train_border = (int)(0.6 * len(x_tmp_divided))
        val_border = train_border + (int)(0.2 * len(x_tmp_divided))
        
        XTrain_d += x_tmp_divided[:train_border]
        XVal_d += x_tmp_divided[train_border:val_border]
        XTest_d += x_tmp_divided[val_border:]

        XTrain_u += x_tmp_united[:train_border]
        XVal_u += x_tmp_united[train_border:val_border]
        XTest_u += x_tmp_united[val_border:]

        YTrain_d += y_tmp_divided[:train_border]
        YVal_d += y_tmp_divided[train_border:val_border]
        YTest_d += y_tmp_divided[val_border:]

        YTrain_u += y_tmp_united[:train_border]
        YVal_u += y_tmp_united[train_border:val_border]
        YTest_u += y_tmp_united[val_border:]

        x_tmp_united.clear()
        x_tmp_divided.clear()
        y_tmp_united.clear()
        y_tmp_divided.clear()
    else:
        random.shuffle(x_tmp_divided)
        train_border = (int)(0.6 * len(x_tmp_divided))
        val_border = train_border + (int)(0.2 * len(x_tmp_divided))
        
        XTrain_d += x_tmp_divided[:train_border]
        XVal_d += x_tmp_divided[train_border:val_border]
        XTest_d += x_tmp_divided[val_border:]

        YTrain_d += y_tmp_divided[:train_border]
        YVal_d += y_tmp_divided[train_border:val_border]
        YTest_d += y_tmp_divided[val_border:]

        x_tmp_divided.clear()
        y_tmp_divided.clear()
        if i % 2 != 0:
            #border_idx_united.append(idx)
            random.shuffle(x_tmp_united)
            train_border = (int)(0.6 * len(x_tmp_united))
            val_border = train_border + (int)(0.2 * len(x_tmp_united))
            
            XTrain_u += x_tmp_united[:train_border]
            XVal_u += x_tmp_united[train_border:val_border]
            XTest_u += x_tmp_united[val_border:]

            YTrain_u += y_tmp_united[:train_border]
            YVal_u += y_tmp_united[train_border:val_border]
            YTest_u += y_tmp_united[val_border:]

            x_tmp_united.clear()
            y_tmp_united.clear()
            
print("archivation")
np.savez_compressed(dirr + "XTest_d.npz", XTest_d)
np.savez_compressed(dirr + "XTest_u.npz", XTest_u)
np.savez_compressed(dirr + "XVal_d.npz", XVal_d)
np.savez_compressed(dirr + "XVal_u.npz", XVal_u)
np.savez_compressed(dirr + "XTrain_d.npz", XTrain_d)
np.savez_compressed(dirr + "XTrain_u.npz", XTrain_u)

np.savez_compressed(dirr + "YTest_d.npz", YTest_d)
np.savez_compressed(dirr + "YTest_u.npz", YTest_u)
np.savez_compressed(dirr + "YVal_d.npz", YVal_d)
np.savez_compressed(dirr + "YVal_u.npz", YVal_u)
np.savez_compressed(dirr + "YTrain_d.npz", YTrain_d)
np.savez_compressed(dirr + "YTrain_u.npz", YTrain_u)