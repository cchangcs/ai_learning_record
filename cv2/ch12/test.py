# encoding:utf-8
import cv2
import numpy as np

img = cv2.imread('j.png', 0)

cv2.imshow('original', img)
print(img.shape)

# 可以将内核看成一个小矩阵，内核在图像上滑动以进行卷积操作，例如模糊，锐化，边缘检测或其他图像处理操作
kernel = np.ones((5, 5), np.uint8)

# 开运算：先腐蚀再膨胀，它可以用来去除噪声
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow('opening', opening)
cv2.moveWindow('opening', x=img.shape[1], y=0)

# 闭运算：先膨胀再腐蚀，用来填充前景物体中的小洞，或者前景物体上的小黑点
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)
cv2.moveWindow('closing', x=img.shape[1] * 2, y=0)

# 形态学梯度
# 其实就是一副图像腐蚀与膨胀的差别
# 结果看上去就像一副前景物体的轮廓
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)
cv2.moveWindow('gradient', x=img.shape[1] * 3, y=0)

# 礼帽
# 原始图像与开运算之后得到的图像的差。
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)
cv2.moveWindow('tophat', x=img.shape[1] * 4, y=0)

# 黑帽  进行闭运算之后得到的图像与原始图像的差
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)
cv2.moveWindow('blackhat', x=img.shape[1] * 5, y=0)

cv2.waitKey(0)
cv2.destroyAllWindows()

