import cv2
import numpy as np

img0 = cv2.imread('pokerQ.jpg')
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
cv2.imshow('pokerQ', img0)

lsd = cv2.createLineSegmentDetector(0)

dlines =lsd.detect(img)
lines = lsd.detect(img)[0]

cv2.waitKey(0)
for dline in dlines[0]:
    x0 = int(round(dline[0][0]))
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))
    cv2.line(img0, (x0, y0), (x1,y1), (0,255,0), 1, cv2.LINE_AA)
    cv2.imshow("LSD", img0)
    cv2.waitKey(10)

cv2.waitKey(0)
cv2.destroyAllWindows()