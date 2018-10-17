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

# 定义命名空间
with tf.name_scope('input'):
    # 定义两个placehlder,None指的是可以是任意的值，根据传入的批次进行确定
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

# 增加隐藏层---优化
# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数-----优化使用交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# loss = tf.reduce_mean(tf.square(y-prediction))
# 梯度下降法-->可以修改学习率，可以更改为其他的优化方法
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# train = tf.train.AdamOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# equal比较两个参数是否是一样的，相同返回True，不同返回False argmax求最大的值在那个位置（比如求预测的概率最大的数字在什么位置）
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率 cast:转换类型，布尔型转换为浮点型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    # 迭代21个周期--->可以增加训练的轮数
    for epoch in range(1):
        # 每个周期一共训练的批次
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('第', str(epoch + 1), '轮准确率：', acc)
