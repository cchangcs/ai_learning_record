# encoding:utf-8
'''
一维直方图：灰度值
二维直方图：饱和度和颜色
使用函数cv2.calcHist()计算直方图既简单又方便，如果绘制颜色直方图的话，需要先将颜色空间从BGR换到HSV控件计算2D直方图
'''

import cv2
import numpy as np

img = cv2.imread('../data/home.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Numpy提供了绘制2D直方图的函数np.histogram2d()
# 绘制1D直方图时我么使用的是np.histogram()

h, s, v = cv2.split(hsv)
hist, xbins, ybins = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]])
pass