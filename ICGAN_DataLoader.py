from __future__ import print_function, division
import scipy
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

from cv2 import imread, resize
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, input_path='input/', output_path='output/', label_path='label.txt', img_res=(218, 178)):
        self.input_path = input_path
        self.output_path = output_path
        self.label_path = label_path
        self.img_res = img_res
        self.label = None
        self.attributes = None

    def load_data(self, batch_size=1, is_testing=False):
        path = glob(self.input_path + '*')
        batch_images = np.random.choice(path, batch_size)
        imgs_in = []
        imgs_out = []
        labels = []
        
        attr_file = open(self.label_path, "r")
        attr = attr_file.readlines()
        picked = [15, 20]
        my_attr = attr[2:]
        if self.label is None:
            self.label = {}
            for l in my_attr:
                ls = l.split()
                self.label[ls[0]] = [int(ls[k+1]) for k in picked]
        if self.attributes is None:
            self.attributes = [attr[1].split()[k] for k in picked]
        for img_path in batch_images:
            name = img_path.split('/')[-1]
            if name[-3:] != 'jpg' and name[-3:] != 'png':
                continue
            img_in = imread(self.input_path + name)[:,:,::-1]
            img_out = imread(self.output_path + name)[:,:,::-1]

            img_in = resize(img_in, self.img_res)
            img_out = resize(img_out, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_in = np.fliplr(img_in)
                img_out = np.fliplr(img_out)

            imgs_in.append(img_in)
            imgs_out.append(img_out)
            labels.append([[self.label[name]]])
            

        imgs_in = np.array(imgs_in)/127.5 - 1.
        imgs_out = np.array(imgs_out)/127.5 - 1.
        labels = np.array(labels)

        return imgs_in, labels, imgs_out
    
    def load_batch(self, batch_size=1, is_testing=False):
        path = glob(self.input_path + '*')
        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_in = []
            imgs_out = []
            labels = []
        
            attr_file = open(self.label_path, "r")
            attr = attr_file.readlines()
            picked = [15, 20]
            my_attr = attr[2:]
            if self.label is None:
                self.label = {}
                for l in my_attr:
                    ls = l.split()
                    self.label[ls[0]] = [int(ls[k+1]) for k in picked]
            for img in batch:
                name = img.split('/')[-1]
                if name[-3:] != 'jpg' and name[-3:] != 'png':
                    continue
                img_in = imread(self.input_path+ name)[:,:,::-1]
                img_out = imread(self.output_path+ name)[:,:,::-1]

                img_in = resize(img_in, self.img_res)
                img_out = resize(img_out, self.img_res)

                # If training => do random flip
                if not is_testing and np.random.random() < 0.5:
                    img_in = np.fliplr(img_in)
                    img_out = np.fliplr(img_out)

                imgs_in.append(img_in)
                imgs_out.append(img_out)
                labels.append([[self.label[name]]])

            imgs_in = np.array(imgs_in)/127.5 - 1.
            imgs_out = np.array(imgs_out)/127.5 - 1.
            labels = np.array(labels)

            yield imgs_in, labels, imgs_out
