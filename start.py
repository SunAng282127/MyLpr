# !/usr/bin/env python
# -*- coding:utf-8 -*-
import pipline as pp
import cv2
import numpy as np

path = './images/7.jpg'


def starting(path):
    """
    车牌识别的入口
    :param path: 图片的路径
    :return: 车牌识别后的号码
    """
    # 读取图片,解码图片
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    # 将读取的图片输入车牌识别的代码中进行处理
    type, res = pp.SimpleRecognizePlate(image)
    # 打印识别结果信息
    print('<--------------------------------------->')
    print('车牌类型为:', type)
    print('车牌号码为:', res[0])
    print('<--------------------------------------->')

    return type, res[0]


if __name__ == '__main__':
    starting(path)
