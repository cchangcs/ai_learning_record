import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret = cap.set(3, 640)
ret = cap.set(4, 480)

cap.read()
cap.read()
cap.read()
cap.read()
cap.read()
cap.read()
cap.read()
cap.read()
cap.read()
cap.read()
cap.read()
cap.read()
cap.read()
cap.read()

frame_no = 100
ret, bgimg0 = cap.read()
bgimg = cv2.cvtColor(bgimg0, cv2.COLOR_BGR2GRAY)
cv2.imshow('bgimg' + str(frame_no), bgimg0)

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    st = cv2.subtract(gray, bgimg)
    ret, threshold = cv2.threshold(st, 50, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contour size:", len(contours))

    img = cv2.drawContours(st, contours, -1, (255, 255, 255), 3)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("subtract", img)
    cv2.imshow('threshold', threshold)

    key = cv2.waitKey(delay=1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite('poker-threshold.jpg', threshold)

cv2.destroyAllWindows()