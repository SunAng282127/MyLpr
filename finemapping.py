# !/usr/bin/env python
# -*- coding:utf-8 -*-
# 精定位算法
import cv2
import numpy as np
import deskew


def fitLine_ransac(pts, zero_add=0):
    if len(pts) >= 2:
        [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((136 - x) * vy / vx) + y)
        return lefty + 30 + zero_add, righty + 30 + zero_add
    return 0, 0


def findContoursAndDrawBoundingBox(image_rgb):
    line_upper = [];
    line_lower = [];

    line_experiment = []
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

    # 在指定的间隔内(-50,0)返回均匀间隔的数字
    for k in np.linspace(-50, 0, 15):
        # 自适应阈值二值化函数根据图片一小块区域的值来计算对应区域的阈值，从而得到也许更为合适的图片。
        binary_niblack = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, k)
        # cv2.imshow("1",binary_niblack)
        # cv2.waitKey(0)

        # 检测物体的轮廓
        imagex, contours, hierarchy = cv2.findContours(binary_niblack.copy(), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bdbox = cv2.boundingRect(contour)
            if (bdbox[3] / float(bdbox[2]) > 0.7 and bdbox[3] * bdbox[2] > 100 and bdbox[3] * bdbox[2] < 1200) or (
                    bdbox[3] / float(bdbox[2]) > 3 and bdbox[3] * bdbox[2] < 100):
                # cv2.rectangle(rgb,(bdbox[0],bdbox[1]),(bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]),(255,0,0),1)
                line_upper.append([bdbox[0], bdbox[1]])
                line_lower.append([bdbox[0] + bdbox[2], bdbox[1] + bdbox[3]])

                line_experiment.append([bdbox[0], bdbox[1]])
                line_experiment.append([bdbox[0] + bdbox[2], bdbox[1] + bdbox[3]])
                # grouped_rects.append(bdbox)

    rgb = cv2.copyMakeBorder(image_rgb, 30, 30, 0, 0, cv2.BORDER_REPLICATE)
    leftyA, rightyA = fitLine_ransac(np.array(line_lower), 3)
    rows, cols = rgb.shape[:2]

    # rgb = cv2.line(rgb, (cols - 1, rightyA), (0, leftyA), (0, 0, 255), 1,cv2.LINE_AA)

    leftyB, rightyB = fitLine_ransac(np.array(line_upper), -3)

    rows, cols = rgb.shape[:2]

    # rgb = cv2.line(rgb, (cols - 1, rightyB), (0, leftyB), (0,255, 0), 1,cv2.LINE_AA)
    pts_map1 = np.float32([[cols - 1, rightyA], [0, leftyA], [cols - 1, rightyB], [0, leftyB]])
    pts_map2 = np.float32([[136, 36], [0, 36], [136, 0], [0, 0]])
    mat = cv2.getPerspectiveTransform(pts_map1, pts_map2)
    image = cv2.warpPerspective(rgb, mat, (136, 36), flags=cv2.INTER_CUBIC)

    # 车牌的角度矫正
    image, M = deskew.fastDeskew(image)
    # cv2.imshow("2", image)
    # cv2.waitKey(0)
    return image
