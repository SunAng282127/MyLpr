# !/usr/bin/env python
# -*- coding:utf-8 -*-
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
import tensorflow as tf
from PIL import Image
import generateds_zh


def train():
    np.random.seed(1337)
    # 获取数据集
    X_train, y_train = generateds_zh.keras_read(num=7403, isTrain=True)
    X_test, y_test = generateds_zh.keras_read(num=4427, isTrain=False)

    # 数据集处理
    X_train = X_train.reshape(-1, 20, 20, 1)
    X_test = X_test.reshape(-1, 20, 20, 1)

    # 搭建网络
    model = Sequential()
    model.add(Convolution2D(
        filters=32,
        kernel_size=(20, 20),
        padding='same',
        input_shape=(20, 20, 1)
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same'
    ))
    model.add(Dropout(0.25))

    model.add(Convolution2D(filters=64,
                            kernel_size=5,
                            padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(filters=512,
                            kernel_size=5,
                            padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # 34 classes
    model.add(Dense(31))
    model.add(Activation('softmax'))

    # 进行优化
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    print('Training---------------------')
    model.fit(X_train, y_train, epochs=15, batch_size=32)

    # 保存
    model.save('./model/model_zh.h5')

    # 测试模型
    print('Testing----------------------')
    loss, accuracy = model.evaluate(X_test, y_test)
    print('test loss:', loss)
    print('test_accuracy:', accuracy)


# 测试真实图片
# 预处理图片
def preprocessing(path):
    img = Image.open(path)
    reIm = img.resize((20, 20), Image.ANTIALIAS)
    img_arr = np.array(reIm.convert('L'))
    threshold = 50
    for i in range(20):
        for j in range(20):
            if (img_arr[i][j] < threshold):
                img_arr[i][j] = 0
            else:
                img_arr[i][j] = 255
    nm_arr = img_arr.reshape([1, 400])
    nm_arr = nm_arr.astype(np.float)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready


def predict(img):
    # 加载模型h5文件
    model = load_model('./model/model_zh.h5')
    # 预测模型
    X = preprocessing(img).reshape(-1, 20, 20, 1)
    Y = np.argmax(model.predict(X), axis=1)
    print(Y)
    return Y


if __name__ == '__main__':
    # train()
    path = "./data_jpg/train_jpg/新/新_4.jpg"
    predict(path)
