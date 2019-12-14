import tensorflow as tf
import DataGen
import os
import argparse

def main():
	LR = 1e-6

    # TODO: E02: Define Linear Regressor graph --> Definition phase (use graph, placeholder, variable, operations)
	graph=tf.Graph()
	with graph.as_default():
		x = tf.compat.v1.placeholder(tf.float32, shape = [], name = "x")
		y = tf.compat.v1.placeholder(tf.float32, shape = [], name = "y")

		#-----------------------FORDWARD PASS-------------------------------
		W = tf.compat.v1.get_variable("W", shape = [], dtype = tf.float32)
		b = tf.compat.v1.get_variable("b", shape = [], dtype = tf.float32)

		z = tf.multiply(x, W)
		y_ = tf.add(z, b)

		loss = (y_- y) ** 2

		#------------------------BACKWARD PASS------------------------------
		d_loss = 2*(y_ - y)
		d_W = x
		d_b = 1

		#------------------------OPTIMIZATION STEP-------------------------
		W_update = W.assign(W - LR * d_loss * d_W)
		b_update = b.assign(b - LR * d_loss * d_b*200)
		#train_op = tf.group(W_update, b_update)



    # TODO: E03: Run a forward pass --> Run phase (use session and the DataDistribution class from previous exercise)

	dataset = DataGen.DataDistribution()

	with tf.compat.v1.Session(graph = graph) as sess:
		sess.run(tf.compat.v1.global_variables_initializer())
		dataset = DataGen.DataDistribution()
		i = 0
		for input_data, label in dataset.generate(num_iters = 10000):
			prediction, loss_val, w_val, b_val = sess.run([y_, loss, W_update, b_update], feed_dict={x: input_data, y: label})
			if i==1000:
				print('[Loss {}], input: {} ---> Prediction {}, Label {}'.format(loss_val, input_data, prediction, label))
				i=0
			i+=1

	print("Ground truth: W = {}, b = {}".format(dataset.W, dataset.b))
	print("Learned: W = {}, b = {}".format(w_val, b_val))

	#Save  Tensorboard
	writer = tf.compat.v1.summary.FileWriter(os.path.expanduser("~/Documents/PostgrauIA/class/summary"), graph = graph)


if __name__ == '__main__':
    main()
