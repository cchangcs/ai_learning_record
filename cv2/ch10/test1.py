import cv2
import numpy as np


img = cv2.imread('messi5.jpg', 0)
rows, cols = img.shape

'''
第一个参数为旋转中心，第二个参数为旋转角度
第三个参数为旋转后的缩放因子
'''
m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.6)

dst = cv2.warpAffine(img, m, (2 * cols, 2 * rows))
cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

