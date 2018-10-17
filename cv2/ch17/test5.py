#encoding:utf-8
import cv2
import numpy as np

org = cv2.imread('../data/cards.png')
imgray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
cv2.imshow('imgray', imgray)

# 白色背景
ret, threshold = cv2.threshold(imgray, 244, 255, cv2.THRESH_BINARY_INV)   # 将黑白颜色反转
cv2.imshow('after threshold', threshold)

image, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = list()
for i, cnt in enumerate(contours):
    areas.append((i, cv2.contourArea(cnt)))

a2 = sorted(areas, key=lambda d: d[1], reverse=True)  # 按面积大小，从小到大排序
cv2.waitKey(0)


for i, are in a2:
    if are < 150:
        continue
    img22 = org.copy()   # 逐个contour显示
    cv2.drawContours(img22, contours, i, (0, 0, 255), 3)
    print(i ,are)

    cv2.imshow('drawContours', img22)
    k = cv2.waitKey(500)
    if k == ord('q'):
        break

# 获取最大或某个contour，剪切
idx = a2[1][0]
mask = np.zeros_like(org)
cv2.drawContours(mask, contours, idx, (0, 255, 0), -1)
out = np.zeros_like(org)
out[mask == 255] = org[mask == 255]
cv2.imshow('out', out)


idx = a2[4][0]
x, y, w, h = cv2.boundingRect(contours[idx])
roi = org[y: y + h, x: x + w]
cv2.imshow('roi', roi)

cv2.waitKey(0)

cv2.destroyAllWindows()