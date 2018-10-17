import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('messi5.jpg', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
# 彩色图像使用 Opencv 加载时是 BGR 模式，但是Matplotlib 是RGB 模式，所以彩色图像如果已经被 opencv读取，它将不会被Matplotlib正确显示

plt.xticks([]), plt.yticks([])
plt.show()