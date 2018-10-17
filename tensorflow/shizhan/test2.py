import tensorflow as tf

x = tf.Variable([1, 2])
a = tf.constant([3, 3])

# 增加一个矩阵的减法op
sub = tf.subtract(x, a)

# 增加一个加法op
add = tf.add(x, a)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
