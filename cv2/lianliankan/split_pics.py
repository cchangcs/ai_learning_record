# encoding:utf-8
'''
图片切割
'''

import cv2
import pickle

filename = 'img/lianliankan2.png'
img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

size = img.shape
print('size:', size)
width = size[1]
height = size[0]

cv2.imshow('src', img)

# 分割为9行8列
x1 = 0
y1 = 0
xp = int(height / 9)
yp = int(width / 8)
mat = []
for x2 in range(xp, height, xp):
    p1 = []
    for y2 in range(yp, width, yp):
        cut = img[x1: x2, y1: y2]
        cv2.imshow('cut', cut)
        cv2.waitKey(100)
        y1 = y2
        p1.append(cut)
    cv2.waitKey(100)
    y1 = 0
    x1 = x2
    mat.append(p1)
'''
pickle提供了一个简单的持久化功能
可以将对象以文件的形式存放在磁盘上
'''
with open('photo_mat', 'wb') as f:
    pickle.dump(mat, f)

print('finish!')

cv2.waitKey(0)
cv2.destroyAllWindows()