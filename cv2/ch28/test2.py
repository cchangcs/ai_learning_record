# encoding:utf-8
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../data/butterfly.jpg', 0)
surf = cv2.xfeatures2d.SURF_create(400)

kp, des = surf.detectAndCompute(img, None)

print(surf.getHessianThreshold())

surf.setHessianThreshold(50000)

kp, des = surf.detectAndCompute(img, None)

img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)

plt.imshow(img2)
plt.show()

print(surf.descriptorSize())

print(surf.getExtended())
surf.setExtended(True)
kp, des = surf.detectAndCompute(img, None)
print(surf.descriptorSize())
print(des.shape)
