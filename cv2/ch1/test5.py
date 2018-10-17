import numpy as np
import cv2


size = (2560, 1600)
# 全黑
black = np.zeros(size)
print(black[34, 56])

cv2.imshow('black', black)
cv2.imwrite('black.jpg', black)

# 全白
black[:] = 255
print(black[34][56])

cv2.imshow('white', black)
cv2.imwrite('white.jpg', black)

cv2.waitKey(0)