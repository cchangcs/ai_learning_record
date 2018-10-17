# encoding:utf-8
'''
膨胀操作会增加图像的白色前景区域，一般在去除噪声时先用腐蚀再用膨胀
因为腐蚀在去除白噪声的同时也会使前景变小，所以需要在进行膨胀操作
噪声已经去除不会再回来，但是前景会增加，膨胀也可以用来连接两个分开的物体
'''

import cv2
import numpy as np

img = cv2.imread('j.png', 0)
cv2.imshow('original', img)
print(img.shape)

kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('dilation', dilation)
cv2.moveWindow('dilation', x=img.shape[1], y=0)

cv2.waitKey(0)
cv2.destroyAllWindows()
