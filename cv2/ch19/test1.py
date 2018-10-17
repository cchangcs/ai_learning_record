# encoding:utf-8
'''
OpenCV中的傅里叶变换-DFT.py
OpenCV中相应的函数是cv2.dft()和cv2.idft()。和前一个得出的结果是一样的，但是是双通道的
第一个通道是结果的实数部分
第二个是结果的虚数部分
输入图像之前先转换成np.float32格式
使用函数cv2.cartToPolar()，它会同时返回幅度和相位
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/messi5.jpg', 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title(122), plt.xticks([]), plt.yticks([])
plt.show()
