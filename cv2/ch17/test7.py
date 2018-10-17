# encoding:utf-8
import cv2
import numpy as np

'''
函数 cv2.convexHull() 可以用来检测一个区县是否具有凸性缺陷，并能纠正缺陷，一般来说凸型曲线
总是凸出来的但至少是平的，如果有地方凹进去了就叫凸性缺失
'''
img = cv2.imread('star.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
image, contours, hierachy = cv2.findContours(thresh, 2, 1)

cnt = contours[0]

hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img, start, end, [0, 255, 0], 0)
    cv2.circle(img, far, 5, [0, 0, 255], -1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
