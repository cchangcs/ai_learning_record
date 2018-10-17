# encoding:utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../data/home.jpg', 0)
# img.ravel() 将图像转化为一维数组，没有中括号
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
