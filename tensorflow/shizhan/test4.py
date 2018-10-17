import tensorflow as tf
# fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    print(sess.run([mul, add]))

# feed
# 创建占位符
input4 = tf.placeholder(tf.float32)
input5 = tf.placeholder(tf.float32)
output = tf.multiply(input4, input5)
with tf.Session() as sess:
    # feed的数据以字典的形式传入
    print(sess.run(output, feed_dict={input4: [5.0], input5: [6.0]}))
