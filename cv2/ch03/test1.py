import numpy as np
import cv2


img = np.zeros((512, 512, 3), np.uint8)

cv2.line(img, pt1=(0, 0), pt2=(511, 511), color=(255, 0, 0), thickness=5)

cv2.arrowedLine(img, pt1=(21, 13), pt2=(151, 401), color=(255, 0, 0), thickness=5)

cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)


cv2.circle(img, center=(447, 63), radius=63, color=(0, 0, 255), thickness=-1)

cv2.ellipse(img, center=(256, 256), axes=(100, 50), angle=0, startAngle=0, endAngle=180, color=255, thickness=1)

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, text='bottomLeftOrigin', org=(10, 400), fontFace=font,
            fontScale=1, color=(255, 255, 255), thickness=1, bottomLeftOrigin=True)
cv2.putText(img, text='OpenCV', org=(10, 500), fontFace=font, fontScale=4,
            color=(255, 255, 255), thickness=2)

cv2.namedWindow("s")
cv2.imshow("s", img)

cv2.imwrite('example2.png', img)

cv2.waitKey(0)
cv2.destroyAllWindows()