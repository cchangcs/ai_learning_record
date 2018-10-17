# encoding:utf-8
'''
OpenCV中的傅里叶变换-逆DFT
在前面的部分实现了一个HPF高通滤波，现在来实现一个LPF低通滤波，
将高频部分去除，其实就是对图像进行模糊操作
首先构建一个掩码将与低频部分对应的地方置为1，
与高频部分对应的区域置为0
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../data/messi5.jpg', 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30: crow + 30, ccol - 30: ccol + 30] = 1
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
