# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/water_coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

'''
距离变换基本含义是计算一个图像中费像素点到最近的零像素点的距离
也就是到零像素点的最短距离
最常见的距离变换算法就是通过连续的腐蚀操作来实现
腐蚀操作的停止条件是所有前景像素都被完全腐蚀
这样根据腐蚀的先后顺序，我们就得到各个前景像素点到前景中心骨架像素点的距离
根据各个像素点的距离值，设置为不同的灰度值，这样就完成了二值图像的距离变换
'''
distance_transform = cv2.distanceTransform(opening, 1, 5)
ret, sure_fg = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)  # 图像相减
cv2.imshow('unknown', unknown)

# 创建标签
ret, markers1 = cv2.connectedComponents(sure_fg)
# 把背景标签为0，其他的使用从1开始的正整数标
markers = markers1 + 1

markers[unknown == 255]  = 0

markers3 = cv2.watershed(img, markers)
img[markers3 == -1] = [255, 0, 0]
cv2.imshow('watershed', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


