# encoding:utf-8
import cv2
import numpy as np

cap = cv2.VideoCapture('../data/vtest.avi')

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    result = fgbg.apply(frame)

    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, (5, 5))
    im, contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        length = cv2.arcLength(c, True)

        if length > 160:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('result', result)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
