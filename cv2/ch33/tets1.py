# encoding:utf-8
import cv2
import numpy as np

cap = cv2.VideoCapture('../data/vtest.avi')
# 可选参数，比如进行建模场景的时间长度， 高斯混合成分的数量-阈值等
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, (5, 5))

    im, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        length = cv2.arcLength(c, True)
        if length > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('mask', fgmask)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
