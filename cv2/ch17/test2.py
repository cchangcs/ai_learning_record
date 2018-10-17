# encoding:utf-8
import cv2
import numpy as np
'''
形状匹配：
函数cv2.matchShape() 可以比较两个形状或轮廓的相似度
如果返回值越小，匹配越高，它是根据 Hu 矩阵来计算的
'''

img1 = cv2.imread('star.jpg', 0)
img2 = cv2.imread('star2.jpg', 0)

ret, thresh = cv2.threshold(img1, 127, 255, 0)
ret, thresh2 = cv2.threshold(img2, 127, 255, 0)

image, contours, hierachy = cv2.findContours(thresh, 2, 1)
cnt1 = contours[0]
image, contours, hierachy = cv2.findContours(thresh2, 2, 1)
cnt2 = contours[0]

ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
print(ret)
