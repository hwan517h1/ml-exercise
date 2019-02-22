import tensorflow as tf

# Constant: a type of tensorflow
hello = tf.constant('Hello, TensorFlow!')
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(c)

# Lazy evaluation: using Session to run graphs
sess = tf.Session()

print(sess.run(hello))
print(sess.run([a, b, c]))
print(sess.run(c))

sess.close()

