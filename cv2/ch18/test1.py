# encoding:utf-8
'''
自适应的直方图均衡化，，这种情况下整幅图像会分成很多小块，这些小块称为titles
在OpenCV中titles 的大小是 8x8
然后再对每一个小块分别做直方图均衡化，
'''
import numpy as np
import cv2

img = cv2.imread('tsukuba_l.png', 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
cv2.imshow('img', img)
cv2.imshow('clahe', cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()
