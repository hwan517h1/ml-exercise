import tensorflow as tf

# Placeholder: a type of tensorflow
X = tf.placeholder(tf.float32, [None, 3])
# X = tf.placeholder(tf.float32) # default
# X = tf.placeholder(tf.float32, [None, None]) # run as default
print(X)

x_data = [[1, 2, 3], [4, 5, 6]]

# Variable: a type of tensorflow
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))

expr = tf.matmul(X, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()

