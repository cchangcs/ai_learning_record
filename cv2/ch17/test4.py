# encoding:utf-8
import cv2
import numpy as np

from pprint import pprint

img2 = cv2.imread('../data/star.png')
img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 127, 255, 0)
cv2.imshow('thresh', thresh)

image, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print('contours length:', len(contours))

cnt = contours[0]
# cv2.moments()会将计算得到的矩以一个字典的形式返回
m = cv2.moments(cnt)

pprint(m)

# 根据这些矩的值，可以计算出对象的重心
cx = int(m['m10'] / m['m00'])
cy = int(m['m10'] / m['m00'])
print('重心：(', cx, ',', cy, ')')

area = cv2.contourArea(cnt)
print('面积：', area)

# 第二个参数可以用来指定对象的形状是闭合的 True ，是打开的 False 一条曲线
perimeter = cv2.arcLength(cnt, True)
print('周长：', perimeter)

'''
将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目有我们定的准确度来决定。
使用 Douglas-Peucker算法
'''
epsilon = 0.1 * cv2.arcLength(cnt, True)
print('epsilon:', epsilon)

approx = cv2.approxPolyDP(cnt, epsilon, True)
cv2.drawContours(img, [approx], 0, (0, 0, 255), 3)
cv2.imshow('approxPolyDP', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
