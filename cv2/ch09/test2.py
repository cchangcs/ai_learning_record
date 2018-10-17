import cv2
import numpy as np


# 蓝色的阈值
lower = np.array([20, 100, 100])
upper = np.array([30, 255, 255])

frame = cv2.imread('ball.jpg')
# 换到HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 根据阈值构建掩码

mask = cv2.inRange(hsv, lower, upper)
# 对原图像和掩码进行位运算
res = cv2.bitwise_and(frame, frame, mask=mask)
# 显示图像
cv2.imshow('frame', frame)
cv2.imshow('mask', mask)
cv2.imshow('res', res)
cv2.waitKey(0)

cv2.destroyAllWindows()
