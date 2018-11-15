from math import exp
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mark = 'network_file'


def sigmoid(x):
    return 1 / (1 + exp(-x))


def get_rms(data1, data2):
    rms = 0
    for y, yh in zip(data1, data2):
        dq = y - yh
        rms += dq * dq
    return np.sqrt(rms / len(data1))


def test_safe(path):
    try:
        f = open(path, 'r')
    except IOError:
        return True
    try:
        return f.readline() == mark + '\n'
    except IOError:
        return False


def convert(array):
    return str(array).replace('[', '{').replace(']', '}').replace(' ', '').replace('e', ' 10^')


class feedforward:

    def __init__(self, *layer_sizes, name='feedforward_network', tf=None, learning_rate=.04, decay_rate=.004,
                 _fromfile=False):
        if tf is None:
            import tensorflow as tef
            self.tf = tef
        else:
            self.tf = tf
        self.__learning_rate = learning_rate
        self.__decay_mul = 1 - decay_rate
        self.__name = name

        self.__initialize_variables()
        if not _fromfile:
            with self.tf.variable_scope(name):
                self._construct(layer_sizes)

    @staticmethod
    def fromfile(path, learning_rate=.04, decay_rate=.004, tf=None):
        f = open(path)
        if f.readline() != mark + '\n':
            raise Exception(path + ' not a neural network file')

        fil = f.readlines()
        f.close()
        for x in range(0, len(fil)):
            fil[x] = fil[x][:-1]

        name = fil[0]
        lens = []

        x = 1
        while fil[x] != '' and x < len(fil):
            lens.append(int(fil[x]))
            x += 1

        f = feedforward(lens, name=name, tf=tf, learning_rate=learning_rate, decay_rate=decay_rate, _fromfile=True)

        weight_matrices = []
        x += 1
        for layers in range(0, len(lens) - 1):
            rows = lens[layers]
            mat = []
            for row in range(0, rows):
                mat.append([])
                for num in fil[x].split(' '):
                    mat[-1].append(float(num))
                x += 1
            weight_matrices.append(mat)
            x += 1
        x += 1

        bias_vectors = []
        for layers in range(0, len(lens) - 1):
            vec = []
            for num in fil[x].split(' '):
                vec.append(float(num))
            bias_vectors.append(vec)
            x += 2

        with f.tf.variable_scope(name):
            f._construct(lens, weight_matrices=weight_matrices, bias_vectors=bias_vectors)
        return f

    def __initialize_variables(self):
        self._input = None
        self._expected = None
        self.__l = []
        self.__b = []
        self._outs = []
        self.__cost = None
        self.__optimize = None
        self._out_z = None

    def _configure_inputs(self, sizes):
        input_ = self.tf.placeholder(self.tf.float32, [None, sizes[0]], name='input')
        expected = self.tf.placeholder(self.tf.float32, [None, sizes[-1]], name='expected_out')
        return input_, expected

    def _construct(s, sizes, weight_matrices=None, bias_vectors=None):
        s._input, s._expected = s._configure_inputs(sizes)
        tin, tex = s._input, s._expected

        s._outs = [tin]
        z = None
        with s.tf.variable_scope('hidden_layers'):
            for x in range(1, len(sizes)):
                with s.tf.variable_scope('hidden_' + str(x)):
                    if weight_matrices is None:
                        s.__l.append(s.tf.Variable(s.tf.random_normal([sizes[x - 1], sizes[x]], stddev=.03),
                                                   name='weight_' + str(x)))
                    else:
                        s.__l.append(s.tf.Variable(weight_matrices[x - 1], name='weight_' + str(x)))
                    if bias_vectors is None:
                        s.__b.append(s.tf.Variable(s.tf.random_normal([sizes[x]]), name='bias_' + str(x)))
                    else:
                        s.__b.append(s.tf.Variable(bias_vectors[x - 1], name='bias_' + str(x)))

                    z = s._w_b_transform(s._outs[-1], s.__l[-1], s.__b[-1])
                if x < len(sizes) - 1:
                    s._outs.append(s.tf.sigmoid(z))

        with s.tf.variable_scope('output_layer'):
            s._outs.append(s.tf.sigmoid(z))

        with s.tf.variable_scope('compute_cost'):
            out = s.tf.clip_by_value(s._outs[-1], 1e-10, .9999999)
            expect = s.tf.clip_by_value(tex, 1e-10, .9999999)
            s.costs = -s.tf.reduce_sum(expect * s.tf.log(out) + (1 - expect) * s.tf.log(1 - out) -
                                    expect * s.tf.log(expect) - (1 - expect) * s.tf.log(1 - expect), axis=1)
            s.__cost = s.tf.reduce_mean(s.costs)

        with s.tf.variable_scope('compute_statistics'):
            out = s._outs[-1]
            # sq = s.tf.subtract(out, s.tf.reshape(tex, s.tf.shape(out)))
            sq = out - tex
            ms = sq * sq
            ms2 = s.tf.reduce_mean(ms, axis=1)
            rms = s.tf.sqrt(ms2)
            s._rms = s.tf.reduce_mean(rms, name='rms')

        with s.tf.variable_scope('Optimizer'):
            s.__optimize = s.tf.train.AdamOptimizer(learning_rate=s.__learning_rate).minimize(s.__cost)

        decays = []
        with s.tf.variable_scope('weight_decay'):
            for xf in range(0, len(s.__l)):
                decays.append(s.tf.assign(s.__l[xf], s.tf.multiply(
                    s.__l[xf], s.__decay_mul, name="decay_" + str(xf + 1))))
        s.__decay = s.tf.group(*decays)

    def _w_b_transform(self, out, weights, biases):
        mul = self.tf.matmul(out, weights)
        return self.tf.add(mul, biases)

    def save(self, sess, path):
        txt = mark + '\n' + self.__name + '\n'
        txt += str(self._input.shape[1]) + '\n'
        for b in self.__b:
            txt += str(b.shape[0]) + '\n'

        for l in self.__l:
            txt += '\n'
            lay = sess.run(l)
            for x in lay:
                for y in x:
                    txt += str(y) + ' '
                txt = txt[:-1]
                txt += '\n'

        for b in self.__b:
            txt += '\n\n'
            bay = sess.run(b)
            for x in bay:
                txt += str(x) + ' '
            txt = txt[:-1]

        if test_safe(path):
            f = open(path, 'w')
            f.write(txt)
            f.close()
        else:
            raise IOError('Do not try to save to a file that already exists')

    def run(self, sess, _input, ex_output):
        return sess.run([self.__optimize, self.__cost, self.__decay, self._rms],
                        feed_dict={self._input: _input, self._expected: ex_output})

    def train(self, sess, inputs, outputs, epochs, batch_size=1, print_=True):
        io = []
        for i, o in zip(inputs, outputs):
            io.append([i, o])
        io = np.array(io)

        costs = []

        # iii = sess.run(self._input_size, feed_dict={self.__input: io[0: batch_size, 0].tolist()})
        # print(iii)
        # exit(0)

        batches = int(len(io) / batch_size)
        for e in range(0, epochs):
            np.random.shuffle(io)
            avg_cost = 0
            avg_rms = 0
            for b in range(0, batches):
                _input = io[b * batch_size:(b + 1) * batch_size, 0].tolist()
                _output = io[b * batch_size:(b + 1) * batch_size, 1].tolist()

                op, cost, decay, rms = self.run(sess, _input, _output)

                avg_rms += rms
                avg_cost += cost
            avg_cost /= batches
            avg_rms /= batches

            costs.append(avg_cost)
            if print_:
                print('epoch ' + str(e + 1) + ':\tcost = {:.3}'.format(avg_cost) + '\tavg rms = {:.3}'.format(avg_rms))
        return costs

    def input(self, sess, inpt):
        res = sess.run(self._outs[-1], feed_dict={self._input: [inpt]})
        return res[0].tolist()
