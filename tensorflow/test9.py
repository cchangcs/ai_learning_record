import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小，一次性放入神经网络的数据数量，以矩阵的形式放入
# 批次优化
batch_size = 100
# batch_size = 100
# 计算一共有多少个批次 //是整除的意思
n_batch = mnist.train.num_examples // batch_size

# 定义两个placehlder,None指的是可以是任意的值，根据传入的批次进行确定
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prop = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)
# 增加隐藏层---优化
# 创建一个简单的神经网络
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# prediction = tf.nn.softmax(tf.matmul(x, W) + b)

W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
l_1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
l1_keep = tf.nn.dropout(l_1, keep_prob=keep_prop)

W2 = tf.Variable(tf.truncated_normal([500, 500], stddev=0.1))
b2 = tf.Variable(tf.zeros([500]) + 0.1)
l_2 = tf.nn.sigmoid(tf.matmul(l1_keep, W2) + b2)
l2_keep = tf.nn.dropout(l_2, keep_prob=keep_prop)

W3 = tf.Variable(tf.zeros([500, 10]))
b3 = tf.Variable(tf.zeros([10]) + 0.1)
# 输出层使用softmax函数做分类
prediction = tf.nn.softmax(tf.matmul(l2_keep, W3) + b3)
# 二次代价函数-----优化使用交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# loss = tf.reduce_mean(tf.square(y-prediction))
# 梯度下降法-->可以修改学习率，可以更改为其他的优化方法
# train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# AdamOptimizer一般使用比较低的学习率（使用1e-6~1e-4等），但是收敛的速度比GradientDecentOptimizer快
train = tf.train.AdamOptimizer(lr).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# equal比较两个参数是否是一样的，相同返回True，不同返回False argmax求最大的值在那个位置（比如求预测的概率最大的数字在什么位置）
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率 cast:转换类型，布尔型转换为浮点型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    # 迭代21个周期--->可以增加训练的轮数
    # for epoch in range(21):
    for epoch in range(100):
        # 每个周期一共训练的批次
        for batch in range(n_batch):
            sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys, keep_prop: 1.0})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prop: 1.0})
        learning_rate = sess.run(lr)
        print('第', str(epoch + 1), '轮准确率：', acc, '  learning rate:', learning_rate)
