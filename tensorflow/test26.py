import tensorflow  as tf

with tf.variable_scope('scope1') as scope:
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
    scope.reuse_variables()
    var1_reuse = tf.get_variable(name='var1')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var1.name)
    print(sess.run(var1))
    print(var1_reuse.name)
    print(sess.run(var1_reuse))
