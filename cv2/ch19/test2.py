# encoding:utf-8
'''
函数 np.fft.fft2() 可以对信号
第一个参数是输入图像，要求是灰度格式
第二个参数是可选的， 决定输出数组的大小
输出数组的大小和输入图像大小是一样的，
如果输出结果比输入图像大，输入图像需要在进行FFT之前补0
如果输出结果比输入图像小的话，输入图像就会被切割
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/messi5.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# 构建振幅
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
