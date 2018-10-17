# encoding:utf-8
import tensorflow as tf
import numpy as np

# 使用numpy生成随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# 构造一个线性模型
b = tf.Variable([0.])
k = tf.Variable([0.])
y = k * x_data + b

# 二次代价函数
# reduce_mean求平均值
# square求平方
loss = tf.reduce_mean(tf.square(y_data - y))

# 定义一个梯度下降法来进行训练的优化器
# 最小化代价函数
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(300):
        sess.run(train)
        if i % 20 == 0:
            print(i, sess.run([k, b, loss]))

