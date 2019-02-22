import tensorflow as tf
import numpy as np

x_data = np.array( # binary classification
		[[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]) # 6 * 2

y_data = np.array([ # one-hot encoding
		[1, 0, 0], # type 1
		[0, 1, 0], # type 2
		[0, 0, 1], # type 3
		[1, 0, 0],
		[1, 0, 0],
		[0, 0, 1]]) # 6 * 3

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([2, 3], -1., 1.)) # 2 is the size of [6, 2], 3 is size of [6, 3] 
b = tf.Variable(tf.zeros([3])) # 3 is the same as the latter of [2, 3]

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)

model = tf.nn.softmax(L) # total sum is 1(possibility)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1)) # cost function(cross-entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # optimizer
train_op = optimizer.minimize(cost)

prediction = tf.argmax(model, axis=1) # find the index of max
target = tf.argmax(Y, axis=1)
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # cast: 0 or 1

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
	sess.run(train_op, feed_dict={X: x_data, Y: y_data})

	if(step + 1) % 10 == 0:
		print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# prediction = tf.argmax(model, axis=1) # find the index of max
# target = tf.argmax(Y, axis=1)
print("Predicted value: ", sess.run(prediction, feed_dict={X: x_data}))
print("Real value: ", sess.run(target, feed_dict={Y: y_data}))

# is_correct = tf.equal(prediction, target)
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # cast: 0 or 1
print("Accuracy: %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))


