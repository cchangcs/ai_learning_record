# encoding:utf-8
import cv2
import numpy as np


'''
检测一副图像中眼睛的位置，应该先在图像中找到脸，然后在脸的区域中找到眼睛，
而不是 直接在一副图像中搜索，这样会提高程序的准确性和性能
'''

img = cv2.imread('messi5.jpg')

ball = img[92: 136, 223: 279]
img[287: 331, 334: 390] = ball  # 修改像素值

cv2.namedWindow('messi', 0)
cv2.imshow('messi', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
