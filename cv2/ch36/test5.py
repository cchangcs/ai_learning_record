# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 生成测试数据
x = np.random.randint(25, 100, 25)
y = np.random.randint(175, 255, 25)
# print(x)
# print(y)

# 矩阵的合并
z = np.hstack((x, y))
# print(z)
# 将横向量变换为列向量
z = z.reshape((50, 1))
# print(z)
z = np.float32(z)
# print(z)
plt.hist(z, 256, [0, 256]), plt.show()

# 使用kmeans进行聚类分析，设置终止条件为执行10次迭代或者精确度epsilon=1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

flags = cv2.KMEANS_RANDOM_CENTERS

# 运用kmeans
# 返回值有紧密度、标志和聚类中心。标志的多少与测试数据的多少是相同的
compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, flags)

A = z[labels == 0]
B = z[labels == 1]

'''
根据数据的标志将数组分为两组，
A组用红色表示
B组用蓝色表示
中心用黄色表示
'''

plt.hist(A, 256, [0, 256], color='r')
plt.hist(B, 256, [0, 256], color='b')
plt.hist(centers, 32, [0, 256], color='y')
plt.show()
