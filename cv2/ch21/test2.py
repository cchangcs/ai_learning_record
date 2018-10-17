import cv2
import numpy as np

img = cv2.imread('../data/sudoku.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100, apertureSize=3)

minLineLenth = 100
maxLineGap = 10

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLenth, maxLineGap=minLineLenth)
print("Len of lines:", len(lines))
print(lines)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('houghlines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()