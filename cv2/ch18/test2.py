# encoding:utf-8
'''
绘制2D直方图
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../data/home.jpg')
hist = cv2.calcHist([img], [0, 1], None, [180, 256], [0, 180, 0, 256])

cv2.imshow('img', img)
cv2.imshow('hist', hist)
cv2.waitKey(0)
cv2.destroyAllWindows()
