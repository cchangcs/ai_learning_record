import cv2
import numpy as np

cap = cv2.VideoCapture(0)
detector = cv2.SimpleBlobDetector_create()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_detect = detector.detect(frame)
    im_blob = cv2.drawKeypoints(frame, im_detect, np.array([]), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame', im_blob)
    key = cv2.waitKey(delay=1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()

