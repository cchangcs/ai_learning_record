# encoding:utf-8
'''
两个形态学操作是腐蚀和膨胀。他们的变体变成了开运算和闭运算
根据卷积核的大小，前景的所有像素会被腐蚀掉变为0， 所以前景物体会变小，整幅图像
的白色区域会被减少
对于去除白噪声很有用，也可以用来断开两个连在一起的物体
'''

import cv2
import numpy as np

img = cv2.imread('j.png', 0)
cv2.imshow('j.png', img)
print(img.shape)

# 可以将内核看成一个小矩阵，在图像上滑动进行（卷积）操作，例如模糊、锐化、边缘检测或其他图像处理擦欧洲哦
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
cv2.imshow('erode', erosion)
cv2.moveWindow('erode', x=img.shape[1], y=0)

cv2.waitKey(0)
cv2.destroyAllWindows()
