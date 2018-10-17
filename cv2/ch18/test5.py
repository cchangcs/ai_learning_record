# encoding:utf-8
'''
OpenCV提供的函数 cv2.calcBackProject() 可以用来做直方图反向投影
它的参数与函数cv2.calcHist() 的参数相同
其中一个参数是 我们查找的目标直方图
在对直方图做方向投影之前，需要先对其进行归一化处理
返回的结果是一个概率图像，然后在使用一个圆盘形卷积核对其进行卷积操作，
最后在进行二值化
'''

import cv2
import numpy as np

roi = cv2.imread('tar.jpg')
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)