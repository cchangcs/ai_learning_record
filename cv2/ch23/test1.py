# encoding:utf-8
'''
一个圆环 需要 3 个参数来确定。所以进行圆环 夫变换的累加器必须是 3 维的
  这样的 效率 就会很低。所以 OpenCV 用来一个比 巧妙的办法 霍夫梯度法 它可以使 用边界的梯度信息。

参数：
image： 8位，单通道图像。如果使用彩色图像，请先转换为灰度。
method：定义检测图像中的圆的方法。目前，唯一实现的方法是cv2.HOUGH_GRADIENT对应于Yuen等。纸。
dp：该参数是累加器分辨率与图像分辨率的反比（详见Yuen等人）。实质上，dp获取越大，累加器数组越小。
minDist：检测到的圆的中心（x，y）坐标之间的最小距离。如果minDist太小，则可能（错误地）检测到与原始相邻的多个圆。如果minDist太大，那么一些圈子根本就不会被检测到。
param1： Yuen等人用于处理边缘检测的梯度值 方法。
param2：该cv2.HOUGH_GRADIENT方法的累加器阈值。阈值越小，检测到的圈子越多（包括虚假圈子）。阈值越大，可能会返回的圈数越多。
minRadius：半径的最小大小（以像素为单位）。
maxRadius：半径的最大大小（以像素为单位）。
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('eye-color-blue-z-c-660x440.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('img'), plt.xticks([]), plt.yticks([])
# hough transform  规定检测的圆的最大最小半径，不能盲目的检测，否则浪费时间空间
# circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=200, maxRadius=300)
circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=100, maxRadius=200)  #把半径范围缩小点，检测内圆，瞳孔
circles = circle1[0, :, :]  # 提取为二维
circles = np.uint16(np.around(circles))  # 四舍五入，取整
for i in circles[:]:
    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)  # 画圆
    cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 10)  # 画圆心

plt.subplot(122), plt.imshow(img)
plt.title('circle'), plt.xticks([]), plt.yticks([])
plt.show()
