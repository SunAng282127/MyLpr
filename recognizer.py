# !/usr/bin/env python
# -*- coding:utf-8 -*-
import codecs
from PIL import Image
from keras.models import *
from keras.layers import *

K.set_image_dim_ordering('tf')
model_ch = load_model('./model/model_ch.h5')
model_zh = load_model('./model/model_zh.h5')
c_Chart = './chart/c_Chart.txt'
z_Chart = './chart/z_Chart.txt'
path = './tem_img/temp.jpg'


# 预处理图片
def preprocessing():
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
    img_ready = np.multiply(nm_arr, 1.0 / 255.0).reshape(-1, 20, 20, 1)

    return img_ready


# 遍历对照表
def traversal_chart(chart_path, chart):
    f = codecs.open(chart_path, 'r', 'utf-8')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split()
        if value[0] == chart:
            return value[1]


def SimplePredict(pos):
    img = preprocessing()
    # 将字符分割后的第一张图片输入到汉字识别模型中进行识别，其他图片输入到另一个模型中
    if pos == 0:
        # 获取参数最大的值的下标则代表识别出来对应的标签
        res = np.argmax(model_zh.predict(img), axis=1)
        # 将标签进行转换，生成对应的汉字
        res1 = traversal_chart(z_Chart, str(res[0]))
    else:
        res = np.argmax(model_ch.predict(img), axis=1)
        res1 = traversal_chart(c_Chart, str(res[0]))
    return res1
