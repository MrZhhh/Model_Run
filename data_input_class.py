# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR10 small images classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import cv2 as cv

from six.moves import cPickle
from tensorflow.python.keras import backend as K
#from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
#from tensorflow.python.util.tf_export import tf_export

HEIGHT_SIZE = 256
WIDTH_SIZE = 256
CHANNELS = 3

class load_data(object):

    @staticmethod
    def load_batch(fpath, label_key='labels'):

        with open(fpath, 'rb') as f:
            if sys.version_info < (3,):
                d = cPickle.load(f)
            else:
                d = cPickle.load(f, encoding='bytes')
            # decode utf8
                d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
                d = d_decoded
            data = d['data']
            labels = d[label_key]

            data = data.reshape(data.shape[0], CHANNELS, HEIGHT_SIZE, WIDTH_SIZE)
            return data, labels

    @staticmethod
    def load_data():
        path = 'D:/Working_Area/Test_data/NUAA/original_data/Face/databatch/'

    # num_train_samples_face = 3474
    # num_train_samples_eye = 3474

    # x1_train = np.empty((num_train_samples_face, 3, 32, 32), dtype='uint8')
    # y1_train = np.empty((num_train_samples_face,), dtype='uint8')
    # x2_train = np.empty((num_train_samples_eye, 3, 32, 32), dtype='uint8')
    # y2_train = np.empty((num_train_samples_eye,), dtype='uint8')

    # for i in range(1, 1):
        # fpath = os.path.join(path, 'data_batch_' + str(i))
        # (x_train[(i - 1) * 10000:i * 10000, :, :, :],
        # y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
        fpath = os.path.join(path, 'databatch_train')
        x_train, y_train = load_data.load_batch(fpath)
        fpath = os.path.join(path, 'databatch_test')
        x_test, y_test = load_data.load_batch(fpath)


        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))
        #y2_test = np.reshape(y2_test, (len(y2_test), 1))

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)
            #x2_test = x2_test.transpose(0, 2, 3, 1)

        return (x_train, y_train), (x_test, y_test)
    
    def LPO(image):
        kernel_size = 3
        scale = 1
        delta = 0
        ddepth = cv.CV_16S
        img = cv.GaussianBlur(image,(3,3),0)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        gray_lap = cv.Laplacian(gray, ddepth, ksize = kernel_size, scale = scale, delta = delta)
        dst = cv.convertScaleAbs(gray_lap)
        
        return dst

    @staticmethod
    def img_process():
        x_out = []
        (x_train, y_train), (x_test, y_test) = load_data.load_data()
        print(x_train.shape)
        num = x_train.shape[0]
        for k in range(0, num):
            image = x_train[k,:,:,:]
            data = load_data.LPO(image)
            data = np.array(data)
            x_out.append(data)
            #print(data.shape)
        x_out = np.array(x_out)
        print(x_out.shape)
        #cv.imwrite("img.jpg", image)
        return x_train, x_out
    
def main():
    load_data.img_process()
    x_train, x_out = load_data.img_process()
    img = x_train[300,:,:,:]
    cv.imwrite("img1.jpg", img)
    img = x_out[300,:,:]
    cv.imwrite("img2.jpg", img)
    
if __name__ == "__main__":
    main()