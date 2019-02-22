import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

hypothesis1 = W1 * X + b1
hypothesis2 = W2 * X + b2

cost1 = tf.reduce_mean(tf.square(hypothesis1 - Y)) # loss function
cost2 = tf.reduce_mean(tf.square(hypothesis2 - Y)) # loss function
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=0.1) # gradient descent
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=0.15) # gradient descent
train_op1 = optimizer1.minimize(cost1)
train_op2 = optimizer2.minimize(cost2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# for step in range(100):
	for step in range(200):
		_, cost_val = sess.run([train_op1, cost1], feed_dict={X: x_data, Y: y_data})
		_, cost_val = sess.run([train_op2, cost2], feed_dict={X: x_data, Y: y_data})
		# cost_val, _ = sess.run([cost, train_op], feed_dict={X: x_data, Y: y_data})
		# print(step, cost_val, sess.run(W), sess.run(b))

	print("\n=== Test1 ===")
	print("X: 5, Y: ", sess.run(hypothesis1, feed_dict={X: 5}))
	print("X: 2.5, Y: ", sess.run(hypothesis1, feed_dict={X: 2.5}))
	print("\n=== Test2 ===")
	print("X: 5, Y: ", sess.run(hypothesis2, feed_dict={X: 5}))
	print("X: 2.5, Y: ", sess.run(hypothesis2, feed_dict={X: 2.5}))
	
