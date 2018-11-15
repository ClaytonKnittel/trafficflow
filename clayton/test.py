from clayton.neural_network import feedforward
import tensorflow as tf

n = feedforward(3, 3, tf=tf, learning_rate=.1, decay_rate=.0)

inp = []
out = []
for x in range(0, 7):
    inp.append([])
    out.append([])
    for i in range(0, 3):
        inp[-1].append((x >> i) & 1)
        out[-1].append(1 - inp[-1][-1])


with tf.Session() as sess:
    init_ops = tf.global_variables_initializer()
    sess.run(init_ops)
    n.train(sess, inp, out, epochs=100)

    print(n.input(sess, [0, 0, 1]))
