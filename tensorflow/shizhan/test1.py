# encoding:utf-8
import tensorflow as tf

# 创建一个常量op
m1 = tf.constant([[3, 3]])
# 创建一个常量op
m2 = tf.constant([[2], [3]])

# 创建一个矩阵乘法op,把m1和m2传入
product = tf.matmul(m1, m2)

with tf.Session() as sess:
    # 调用sess的run方法来执行矩阵的乘法op
    # run(product)触发了图中三个op
    print(sess.run(product))
