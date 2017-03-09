import tensorflow as tf
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[1, 6], name='a')
c = tf.reduce_mean(tf.cast(a,'float32'))
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print sess.run(c)
