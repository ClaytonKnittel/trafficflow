import tensorflow as tf
import numpy as np

inpt = tf.placeholder(tf.float32, [8], name='input')
inpt2 = tf.reshape(inpt, [8, 1])

w1 = tf.Variable(tf.random_normal([3, 8], stddev=.3), name='weight1')

encoding = tf.matmul(w1, inpt2)
encoding = tf.sigmoid(encoding)

w2 = tf.Variable(tf.random_normal([8, 3], stddev=.3), name='weight2')

res = tf.sigmoid(tf.matmul(w2, encoding))

inpt2 = tf.clip_by_value(inpt2, 0.001, 0.999)
res = tf.clip_by_value(res, .001, .999)

cross_entropy = tf.negative(tf.add(tf.multiply(inpt2, tf.log(res)), tf.multiply(tf.negative(tf.subtract(inpt2, 1)), tf.log(tf.negative(tf.subtract(res, 1))))))
n_cross_entropy = tf.add(cross_entropy, tf.add(tf.multiply(inpt2, tf.log(inpt2)), tf.multiply(tf.negative(tf.subtract(inpt2, 1)), tf.log(tf.negative(tf.subtract(inpt2, 1))))))

errvec = tf.reshape(n_cross_entropy, [-1])
err = tf.reduce_mean(errvec)
cost = err

opt = tf.train.AdamOptimizer(learning_rate=.5).minimize(cost)

def gen_data(size):
    data = []
    for x in range(0, size):
        data.append(x)
    return data


size = 8

data = gen_data(size)

#tf.reset_default_graph()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # fw = tf.summary.FileWriter('./logs/traffic/train', sess.graph)

    for x in range(0, 100):
        eavg = 0
        for d in data:
            inp = [0] * size
            inp[d] = 1
            o, e = sess.run([opt, err], feed_dict={inpt: inp})
            eavg += e
        # print(eavg / len(data))


    for x2 in data:
        ar = [0] * size
        ar[x2] = 1
        i = sess.run(res, feed_dict={inpt: ar})
        print(ar, '\n', i, '\n')

    # fw.close()

