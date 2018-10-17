import numpy as np
import cv2

img = cv2.imread('../data/home.jpg')

Z = img.reshape((-1, 3))

Z = np.float32(Z)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 8

ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 分离颜色
for y in range(len(center)):
    a1 = []
    for i, x in enumerate(label.ravel()):
        if x == y:
            a1.append(list(Z[i]))
        else:
            a1.append([0, 0, 0])
    a2 = np.array(a1)
    a3 = a2.reshape((img.shape))
    cv2.imshow('res2' + str(y), a3)

cv2.waitKey(0)
cv2.destroyAllWindows()