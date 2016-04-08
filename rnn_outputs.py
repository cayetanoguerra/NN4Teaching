"""
Simple demo of an oscillator net
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def tanh(x):
    return 2./(1. + np.exp(-2. * x)) - 1.


def indentity(x):
    return x


class Layer:
    """ Layer
    """

    def __init__(self, rows, columns, activ_f=sigmoid):
        self.rows = rows
        self.columns = columns
        self.W = Layer.create_W(self.rows, self.columns)
        self.activ_f = activ_f

    @staticmethod
    def create_W(rows, columns):
        return np.random.rand(rows, columns)*2. - 1.

    def output(self, x):
        self.o = self.activ_f(np.dot(self.W, x))
        return self.o

    def __str__(self):
        return "Layer_____________\n     #inputs:" + repr(self.columns) + "\n     #neurons: " + repr(self.rows)


neurons = 5
inputs = 2

hidden_layer = Layer(neurons, neurons + inputs, activ_f=indentity)
output_layer = Layer(1, neurons, activ_f=tanh)

original_inputs = np.zeros(neurons)
inputs = np.asarray([1., 1.])

complete_input = np.append(inputs, original_inputs)

y = np.asarray([])

for i in xrange(100):

    if i < 50:
        inputs = np.zeros(2)
        #inputs = np.random.rand(2)
    elif i < 100:
        inputs = np.zeros(2)
    else:
        inputs = np.ones(2)

    h_o = hidden_layer.output(complete_input)
    complete_input = np.append(inputs, h_o)
    print "Hidden layer: ", h_o
    o = output_layer.output(h_o)
    y = np.append(y, h_o)

y = y.reshape((100, 5))

plt.plot(y[:, 0], 'b')
plt.plot(y[:, 1], 'r')
plt.plot(y[:, 2], 'g')
plt.plot(y[:, 3], 'y')
plt.plot(y[:, 4], 'k')

plt.show()

