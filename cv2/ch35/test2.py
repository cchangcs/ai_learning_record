import cv2
import numpy as np
import matplotlib.pyplot as plt

# 左边摄像头拍到的图
imgL = cv2.imread('tsukuba_l.png', 0)
# 右边摄像头拍到的图
imgR = cv2.imread('tsukuba_r.png', 0)

# 立体匹配算法
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()
