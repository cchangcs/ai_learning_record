# encoding:utf-8
import cv2
import numpy as np, sys
import matplotlib.pyplot as plt


def sameSize(img1, img2):
    rows, cols, dpt = img2.shape
    dst = img1[:rows,:cols]
    return dst

apple = cv2.imread('apple.jpg')
orange = cv2.imread('orange.jpg')

# generate gaussian pyramid for apple
g = apple.copy()
gp_apples = [g]
for i in range(6):
    g = cv2.pyrDown(g)
    gp_apples.append(g)


g = orange.copy()
gp_oranges = [g]
for i in range(6):
    g = cv2.pyrDown(g)
    gp_oranges.append(g)

# generate Laplacian Pytamid for apple
lp_apples = [gp_apples[5]]
for i in range(5, 0, -1):
    ge = cv2.pyrUp(gp_apples[i])
    l = cv2.subtract(gp_apples[i - 1], sameSize(ge, gp_apples[i - 1]))
    lp_apples.append(l)

# generate Laplacian Pyramid for B
lp_oranges = [gp_oranges[5]]
for i in range(5, 0, -1):
    ge = cv2.pyrUp(gp_oranges[i])
    l = cv2.subtract(gp_oranges[i - 1], sameSize(ge, gp_oranges[i - 1]))
    lp_oranges.append(l)

ls1 = []
for la, lb in zip(lp_apples, lp_oranges):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0: int(cols / 2)], lb[:, int(cols / 2):]))
    ls1.append(ls)

ls2 = ls1[0]
for i in range(1, 6):
    ls2 = cv2.pyrUp(ls2)
    ls2 = cv2.add(sameSize(ls2, ls1[i]), ls1[i])

rows, cols, ch = apple.shape
real = np.hstack((apple[:, : int(cols / 2)], orange[:, int(cols / 2):]))

plt.subplot(221), plt.imshow(cv2.cvtColor(apple, cv2.COLOR_BGR2RGB))
plt.title("apple"), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(cv2.cvtColor(orange, cv2.COLOR_BGR2RGB))
plt.title("orange"), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(cv2.cvtColor(real, cv2.COLOR_BGR2RGB))
plt.title("real"), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(cv2.cvtColor(ls2, cv2.COLOR_BGR2RGB))
plt.title("ls"), plt.xticks([]), plt.yticks([])
plt.show()
