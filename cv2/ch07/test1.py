import numpy as np
import cv2

x = np.uint8([250])
y = np.uint8([10])
print(cv2.add(x, y))

print(x + y)

# 图像混合
img1 = cv2.imread('ml.png')
img2 = cv2.imread('opencv_logo.png')

img1 = cv2.imread('ml.png')
img2 = cv2.imread('opencv_logo.png')


dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)  # 第一幅图的权重是 0.7 第二幅图的权重是 0.3

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
