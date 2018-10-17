# encoding:utf-8
import cv2

img = cv2.imread('../data/home.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, img)

# 计算关键点描述符
# 使用函数 sift.compute() 来计算这些关键点的描述符
kp, des = sift.detectAndCompute(gray, None)

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
