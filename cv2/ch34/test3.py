# encoding:utf-8
import numpy as np
import cv2
import glob

# 设置终止条件，迭代30次或移动0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 准备对象点，类似（0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, : 2] = np.mgrid[0: 7, 0: 6].T.reshape(-1, 2)  # np.mgrid()返回多维结构

# 从所有图像中存储对象点和图像点的数组
objpoints = []  # 真实世界的3D点
imgpoints = []  # 图像的2D点
# glob.globglob.glob函数的参数是字符串。这个字符串的书写和我们使用
# linux的shell命令相似，或者说基本一样。也就是说，只要我们按照平常使
# 用cd命令时的参数就能够找到我们所需要的文件的路径。字符串中可以包括“*”、
# “?”和"["、"]"，其中“*”表示匹配任意字符串，“?”匹配任意单个字符，
# [0-9]与[a-z]表示匹配0-9的单个数字与a-z的单个字符。
images = glob.glob('../data/left*.jpg')
images += glob.glob('../data/right*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘边界，角点检测
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    # 如果找到，则添加对象点和图像点
    if ret == True:
        objpoints.append(objp)
        # 亚像素级角点检测，在角点检测中精确化角点位置
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # 绘制并展示边界
#         cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
#         print(objpoints)
#         print(imgpoints)
#         cv2.imshow('img', img)
#         cv2.waitKey(500)
# cv2.destroyAllWindows()
# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:: -1], None, None)

np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# 畸形校正
img = cv2.imread('../data/left12.jpg')
cv2.imshow('source', img)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 使用cv2.undistort()，和上面得到的ROI对结果进行剪裁
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# 裁剪图像
x, y, w, h = roi
dst = dst[y: y + h, x: x + w]
cv2.imshow('re1', dst)

# 找到畸变图像到畸变图像的映射方程，在使用重映射方程
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# 裁剪图像
x, y, w, h = roi
dst = dst[y: y + h, x: x + w]
cv2.imshow('re2', dst)
'''
反向投影误差，我们可以利用反向投影误差对我们找到的参数的准确性评估，
得到的结果越接近0越好，有了内部参数、畸变参数和旋转变化矩阵，
就可以使用cv2.projectPoints()将对象转换到图像点
然后就可以计算变换得到的图像与角点检测算法的绝对差了
最后计算所有标定图像的误差平均值
'''
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: ", mean_error / len(objpoints))

cv2.waitKey(0)
cv2.destroyAllWindows()
