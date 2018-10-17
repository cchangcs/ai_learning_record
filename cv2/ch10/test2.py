import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('sudoku.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

m = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, m, (300, 300))

plt.figure(figsize=(8, 7), dpi=98)
p1 = plt.subplot(211)
p1.imshow(img)
p1.set_title('Input')

p2 = plt.subplot(212)
p2.imshow(dst)
p2.set_title('Output')

plt.show()
