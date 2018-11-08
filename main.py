import tensorflow as tf
import numpy as np

inpt = tf.placeholder(tf.float32, [8], name='input')
inpt2 = tf.reshape(inpt, [8, 1])

w1 = tf.Variable(tf.random_normal([3, 8], stddev=.03), name='weight1')

encoding = tf.matmul(w1, inpt2)

w2 = tf.Variable(tf.random_normal([8, 3], stddev=.03), name='weight2')

res = tf.matmul(w2, encoding)

err = tf.reduce_mean(tf.square(tf.subtract(res, inpt)))

opt = tf.train.AdamOptimizer(learning_rate=.03).minimize(err)

init_op = tf.global_variables_initializer()


def gen_data(size):
    data = []
    # len = int(np.ceil(np.log2(size)))
    for x in range(0, size):
        # put = [x]
        # bin = []
        # for y in range(0, len):
        #     bin = [((x >> y) & 1)] + bin
        # put.append(bin)
        data.append(x)
    return data


size = 8

data = gen_data(size)

with tf.Session() as sess:
    sess.run(init_op)

    for x in range(0, 100):
        eavg = 0
        for d in data:
            inp = [0] * size
            inp[d] = 1
            o, e = sess.run([opt, err], feed_dict={inpt: inp})
            eavg += e
        print(eavg / len(data))

    i = sess.run(res, feed_dict={inpt: [0, 0, 0, 0, 0, 0, 0, 1]})
    print(i)

