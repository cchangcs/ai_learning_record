import cv2
import numpy as np

img = cv2.imread('../data/messi5.jpg', 0)
edges = cv2.Canny(img, 150, 200)

cv2.imshow('original', img)

cv2.imshow('edge', edges)
cv2.moveWindow('edge', x=img.shape[1], y=0)

cv2.waitKey(0)
cv2.destroyAllWindows()
