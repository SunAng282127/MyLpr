# !/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2

watch_cascade = cv2.CascadeClassifier('./model/cascade.xml')


def computeSafeRegion(shape, bounding_rect):
    top = bounding_rect[1]  # y
    bottom = bounding_rect[1] + bounding_rect[3]  # y +  h
    left = bounding_rect[0]  # x
    right = bounding_rect[0] + bounding_rect[2]  # x +  w

    min_top = 0
    max_bottom = shape[0]
    min_left = 0
    max_right = shape[1]

    if top < min_top:
        top = min_top

    if left < min_left:
        left = min_left

    if bottom > max_bottom:
        bottom = max_bottom

    if right > max_right:
        right = max_right


    return [left, top, right - left, bottom - top]


def cropped_from_image(image, rect):
    x, y, w, h = computeSafeRegion(image.shape, rect)
    return image[y:y + h, x:x + w]


def detectPlateRough(image_gray, resize_h=720, en_scale=1.08, top_bottom_padding_rate=0.05):
    """
    标记图像中所有车牌的边框在图片中的bbox
    :param image_gray:
    :param resize_h:重新设定的图像大小
    :param en_scale:
    :param top_bottom_padding_rate:表示要裁剪掉图片的上下部占比
    :return:一个表示车牌区域坐标边框的list
    """
    if top_bottom_padding_rate > 0.2:
        print("error:top_bottom_padding_rate > 0.2:", top_bottom_padding_rate)
        exit(1)

    # 根据长宽比重新裁剪图片并按照比例将图片的下半部分裁剪掉
    height = image_gray.shape[0]
    padding = int(height * top_bottom_padding_rate)
    scale = image_gray.shape[1] / float(image_gray.shape[0])
    image = cv2.resize(image_gray, (int(scale * resize_h), resize_h))

    # 裁剪掉top_bottom_padding_rate比例的垂直部分
    image_color_cropped = image[padding:resize_h - padding, 0:image_gray.shape[1]]
    # cv2.imshow('1', image_color_cropped)
    # cv2.waitKey(0)
    # 裁剪之后的图片进行灰度化处理
    image_gray = cv2.cvtColor(image_color_cropped, cv2.COLOR_RGB2GRAY)
    # 根据前面的cv2.CascadeClassifier()物体检测模型(3)，输入image_gray灰度图像，边框可识别的最小size，最大size，
    # 输出得到车牌在图像中的offset，也就是边框左上角坐标( x, y )以及边框高度( h )和宽度( w )
    watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))
    # 对得到的车牌边框的bbox进行扩大(此刻得到的车牌可能因为车牌倾斜等原因导致显示不完整)，
    # 先对宽度左右各扩大0.14倍，高度上下各扩大0.6倍
    cropped_images = []
    for (x, y, w, h) in watches:
        cropped_origin = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))
        x -= w * 0.14
        w += w * 0.28
        y -= h * 0.6
        h += h * 1.1
        # 按扩大之后的大小进行裁剪
        cropped = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))
        cropped_images.append([cropped, [x, y + padding, w, h], cropped_origin])
    # cv2.waitKey(0)
    # 返回的内容包括扩大裁剪后车牌区域的图像、车牌大概的位置以及未扩大后的车牌图像
    return cropped_images
