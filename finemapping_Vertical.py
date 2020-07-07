# !/usr/bin/env python
# -*- coding:utf-8 -*-
from keras.layers import Conv2D, Input, MaxPool2D, Activation, Flatten, Dense
from keras.models import Model
import numpy as np
import cv2

'''
keras网络模型：对车牌的左右边界进行回归 
通过modelFineMapping.loadweights()函数加载模型文件 
通过modelFineMapping.predict输出网络结果

输入：16*66*3 tensor 
输出：长度为2的tensor
'''


def getModel():
    input = Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = Activation("relu", name='relu1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = Activation("relu", name='relu2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = Activation("relu", name='relu3')(x)
    x = Flatten()(x)
    output = Dense(2, name="dense")(x)
    output = Activation("relu", name='relu4')(output)
    model = Model([input], [output])
    return model


model = getModel()
model.load_weights("./model/model12.h5")


def finemappingVertical(image):
    resized = cv2.resize(image, (66, 16))
    resized = resized.astype(np.float) / 255
    res = model.predict(np.array([resized]))[0]
    # print("keras_predict", res)
    res = res * image.shape[1]
    res = res.astype(np.int)
    H, T = res
    H -= 3

    if H < 0:
        H = 0
    T += 2;

    if T >= image.shape[1] - 1:
        T = image.shape[1] - 1

    image = image[0:35, H:T + 2]

    image = cv2.resize(image, (int(136), int(36)))
    return image
