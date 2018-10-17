# encoding:utf-8
import cv2
import numpy as n

'''
使用OpenCV检测程序性能
'''

img1 = cv2.imread('ml.png')

cv2.setUseOptimized(False)
print(cv2.useOptimized())
e1 = cv2.getTickCount()
for i in range(5, 49, 2):
    img1 = cv2.medianBlur(img1, i)

e2 = cv2.getTickCount()

t = (e2 - e1) / cv2.getTickFrequency()  # 时钟频率 或者 每秒钟的时钟数
print(t)
cv2.setUseOptimized(True)
print(cv2.useOptimized())
e1 = cv2.getTickCount()
for i in range(5, 49, 2):
    img1 = cv2.medianBlur(img1, i)

e2 = cv2.getTickCount()

t = (e2 - e1) / cv2.getTickFrequency()  # 时钟频率 或者 每秒钟的时钟数
print(t)