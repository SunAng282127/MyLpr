# !/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import detect
import finemapping as fm
import finemapping_Vertical as fv
import segmentation
import ctype

type = ['蓝牌', '单层黄牌']


def SimpleRecognizePlate(image):
    """
    调用车牌识别所使用的各种步骤
    :param image: 读取的图片
    :return: 返回所有识别出来的车牌号码
    """
    # detectPlateRough是返回图像中所有车牌的边框在图片中的bbox，返回的是一个表示车牌区域坐标边框的list
    images = detect.detectPlateRough(image, image.shape[0], top_bottom_padding_rate=0.1)
    # 将识别出来所有的车牌号码存为一个list
    res_set = []
    for j, plate in enumerate(images):
        plate, rect, origin_plate = plate
        plate = cv2.resize(plate, (136, 36 * 2))

        ptype = ctype.get_color(plate)
        if ptype == 1:
            plate = cv2.bitwise_not(plate)

        # 精确定位，倾斜校正等
        image_rgb = fm.findContoursAndDrawBoundingBox(plate)

        # 对车牌的左右边界进行回归
        image_rgb = fv.finemappingVertical(image_rgb)

        # 车牌区域灰度化
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 基于滑动窗口的字符分割与识别
        val = segmentation.slidingWindowsEval(image_gray)
        res_set.append(val)
        # print("车牌:", val)
    return type[ptype], res_set
