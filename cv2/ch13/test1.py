# encoding:utf-8
'''
2D 卷积
OpenCV提供的函数 cv.filter2D() 可以对任何一副图像进行卷积操作
操作如下：
将卷积核放在图像的一个像素A上，求与核对应的图像上 25 5x5 个像素的和，再取平均数
用这个平均数替代像素A的值，重复以上的操作直到将图像的每一个像素更新
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/opencv_logo.png')
kernel = np.ones((5, 5), np.float32) / 25

# print(kernel)
dst = cv2.filter2D(img, -1, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

