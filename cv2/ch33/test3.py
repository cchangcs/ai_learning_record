import numpy as np
import cv2
cap = cv2.VideoCapture('../data/vtest.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

counter = 0
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame', fgmask)  # 前120帧
    counter += 1
    print(counter)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
