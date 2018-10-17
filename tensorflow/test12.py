# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次 // 代表整除
n_batch = mnist.train.num_examples // batch_size


# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个数据的正态分布，其中stddev是标准差
    return tf.Variable(initial_value=initial)


# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, W):
    '''
    x:shape[batch, in_height, in_width, in_channels]
    strides:strides[0] = strides[3] = 1,strides[1]代表x方向的步长，strides[2]代表y方向的步长
    padding：'SAME' 'VALID'
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    '''
    x:shape[batch, in_height, in_width, in_channels]
    ksize：ksize[0]=ksize[3]=1,ksize[1]和ksize[2]代表窗口的大小，可以改变
    strides:strides[0] = strides[3] = 1,strides[1]代表x方向的步长，strides[2]代表y方向的步长
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])  # 28x28
y = tf.placeholder(tf.float32, [None, 10])

# 改变x的格式为4D的向量 [batch, in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积核和偏置
W_conv1 = weight_variable([5, 5, 1, 32])  # 5x5的采样窗口，32个卷积核从1个平面抽取特征,使用32个卷积核可以得到32个特征平面
b_conv1 = bias_variable([32])   # 每个卷积核一个偏置值

# 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 初始化第二个卷积核的权值和偏执
W_conv2 = weight_variable([5, 5, 32, 64])  # 5x5的采样窗口，64个卷积核从32个平面抽取特征
b_conv2 = bias_variable([64])  # 每个卷积核一个偏置项

# 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
28x28的图片第一次卷积之后28x28，第一次池化后14x14
第二次卷积后14x14，第二次池化后7x7
通过上面的操作可以得到64张7x7的平面
'''

# 初始化一个全连接层的权值
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # 上一层一共有7x7x64个神经元，全连接层1024个神经元
b_fc1 = bias_variable([1024])

# 将池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第一个全连接层
W_fc2 = weight_variable([1024, 10])  # 上一层一共有7x7x64个神经元，全连接层1024个神经元
b_fc2 = bias_variable([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用AdamOptimizer进行优化
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 初始化参数
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print('Iter ' + str(epoch) + ' 准确率：', acc)
