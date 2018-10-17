# encoding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy来生成200个随机点
# 从-0.5到0.5产生200个随机点
# 生成200行1列的数据
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
'''
生成正态分布的随机点
loc：float
    此概率分布的均值（对应着整个分布的中心centre）
scale：float
    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints
    输出的shape，默认为None，只输出一个值
'''
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder
# placeholder的定义需要根据样本的形状
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_l1 = tf.matmul(x, Weights_L1) + biases_L1

L1 = tf.nn.tanh(Wx_plus_b_l1)
# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_l2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_l2)

# 定义二次代价函数
loss = tf.reduce_mean(tf.square(prediction - y_data))
# 定义梯度下降算法训练
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化所有的变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(2000):
        sess.run(train, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    # 散点图
    plt.scatter(x_data, y_data)
    # 'r-'红色实线
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
