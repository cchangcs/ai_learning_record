# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 将图片切割为5000个部分，每个部分20 x 20
# vsplit沿着垂直轴分割，hsplit沿着水平轴分割
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

# 将cells转化为Numpy数组，大小为(50, 100, 20, 20)
x = np.array(cells)
# 准备测试数据和训练数据
train = x[:, : 50].reshape(-1, 400).astype(np.float32)  # size=(2500, 400)
test = x[:, 50: 100].reshape(-1, 400).astype(np.float32)  # size = (2500, 400)

# 对数组中的每一个元素进行复制
# 除了待重复的数组之外，只有一个额外的参数时，高维数组也会 flatten 至一维
# 创建labels
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

# 初始化KNN， 训练训练数据，并设置k=1来测试测试数据
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=5)

# 计算分类的准确率
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print('准确率：', accuracy)

np.savez('knn_data_num.npz', train=train, train_labels=train_labels, test=test, test_labels=test_labels)

with np.load('knn_data_num.npz') as data:
    print(data.files)
    train = data['train']
    train_labels = data['train_labels']
    test = data['test']
    test_labels = data['test_labels']

# 对数字进行预测
retval, results = knn.predict(test[1003:1005])
# Docstring: predict(samples[, results[, flags]]) -> retval, results
print(retval, results)  # (4.0, array([[ 4.],[ 4.]], dtype=float32))
# 对比
cv2.imshow('test', test[1005].reshape((20, 20)))
cv2.waitKey(0)
cv2.destroyAllWindows()

