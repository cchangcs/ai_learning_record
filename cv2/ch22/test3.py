import cv2
import numpy as np

im = cv2.imread('blob.jpg', cv2.IMREAD_GRAYSCALE)

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 200

params.filterByArea = True
params.minArea = 1500

params.filterByCircularity = True
params.minConvexity = 0.87

params.filterByInertia = True
params.minInertiaRatio = 0.01

ver = (cv2.__version__).split('.')
if(int(ver[0])) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(im)

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('im_with_keypoints', im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
