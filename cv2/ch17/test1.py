# encoding:utf-8
'''
cv2.findContours() 可以 用来绘制轮廓，
它可以根据提供图像的边界点绘制任何形状
第一个参数是原始图像
第二个参数是 轮廓（一个python列表）
第三个参数是 轮廓的索引
在绘制独立的轮廓时很有用，设置为-1时绘制所有轮廓
接下来的参数是轮廓的颜色和厚度
'''

import numpy as np
import cv2

im = cv2.imread('../data/cards.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', imgray)

ret, threshold = cv2.threshold(imgray, 244, 255, 0)
img, contours, hierachy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print('len(contours)', len(contours))

contours2 = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
print('过滤太小的contout:', len(contours2))

cv2.drawContours(im, contours, -1, (255, 0, 0), 3)

if len(contours) > 4:
    # 绘制独立的轮廓
    cv2.drawContours(image=im, contours=contours, contourIdx=3, color=(0, 0, 255), thickness=3)
    cnt = contours[4]
    cv2.drawContours(im, [cnt], 0, (0, 255, 0), 3)

print('contours[0]:', contours[0])
cv2.drawContours(imgray, contours[0], 0, (0, 0, 255), 3)

cv2.imshow('drawContours', im)
cv2.imshow('drawContours-', imgray)
cv2.waitKey(0)
cv2.destroyAllWindows()
