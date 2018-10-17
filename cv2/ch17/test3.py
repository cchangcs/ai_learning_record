# encoding:utf-8
import numpy as np
import cv2

im = cv2.imread('../data/chessboard.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

cv2.imshow('imgray', imgray)

# 需要注意的是cv2.findContours() 函数接受的是二值图，即黑白的（不是灰度图）
# 所以首先需要将图像转化为灰度图,再转成二值图
ret, thresh = cv2.threshold(src=imgray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

cv2.imshow('thresh', thresh)

# 轮廓提取模式
image, contours, hirerchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print('contours size:', len(contours))

img = cv2.drawContours(im, contours, -1, (0, 0, 255), 3)

cv2.imshow('contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

