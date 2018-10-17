import numpy as np
import matplotlib.pyplot as plt

'''
N:batch_size
D_in:输入维度
H:隐藏层维度
D_out:输出维度
'''
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机输入和输出值
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# print(x.shape)
# print(y.shape)

# 随机初始化权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
# print(w1.shape, w2.shape)

learning_rate = 1e-6
for t in range(500):
    # 前向预测网络输出值
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 计算并输出损失值
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 反向计算w1和w2的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新权值
    w1 -= learning_rate * w1
    w2 -= learning_rate * w2

