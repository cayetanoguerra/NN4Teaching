"""
Simple demo of an oscillator net
"""

import numpy as np
import matplotlib.pyplot as plt


#
# Sigmoid
#
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_derivate(o):
    return o * (1.0 - o)


#
# Rectified Linear Unit
#
def relu(x):
    if x >= 0.:
        return x
    else:
        return 0.


def relu_derivate(o):
    if o >= 0.:
        return 1
    else:
        return 0.


#
# Identity
#
def identity(x):
    return x


def identity_derivate():
    return 1


#
# TanH
#
def tanh(x):
    return 2/(1 + np.exp(-2*x)) - 1


def tanh_derivate(o):
    return 1 - o**2


activation_function = {
    "Sigmoid": (sigmoid, sigmoid_derivate),
    "ReLU": (relu, relu_derivate),
    "Identity": (identity, identity_derivate),
    "TanH": (tanh, tanh_derivate)
}


class Layer:
    """ Layer
    """

    def __init__(self, rows, columns, activ_f=activation_function["Sigmoid"]):
        self.rows = rows
        self.columns = columns + 1  # One column for bias
        self.W = self.create_W(self.rows, self.columns)
        self.W_delta = []
        self.b_delta = []
        self.activ_f = activ_f[0]
        self.o_activ_f_derivate = activ_f[1]
        self.error = []
        self.o = None

    def create_W(self, rows, columns):
        return np.random.rand(rows, columns) * 0.1

    def output(self, x):
        x = np.append(x, 1.0)
        self.o = self.activ_f(np.dot(self.W, x))
        self.o_derivate = self.o_activ_f_derivate(self.o)
        return self.o

    def __str__(self):
        return "Layer_____________\n     #inputs:" + repr(self.columns-1) + "\n     #neurons: " + repr(self.rows)


class RNNLayer:
    """ Layer
    """

    def __init__(self, number_of_neurons, external_inputs, activ_f=activation_function["Sigmoid"]):
        self.neurons = number_of_neurons
        self.external_inputs = external_inputs
        self.W = self.create_W(self.neurons, self.neurons + self.external_inputs)
        self.activ_f = activ_f[0]
        self.o_activ_f_derivate = activ_f[1]
        self.error = []
        self.o = None

    def create_W(self, rows, columns):
        return np.random.rand(rows, columns) * 10. - 5.5

    def output(self, x):
        self.o = self.activ_f(np.dot(self.W, x)) #x first neuron outputs, after that external inputs.
        self.o_derivate = self.o_activ_f_derivate(self.o)
        return self.o

    def __str__(self):
        return "Layer_____________\n     #inputs:" + repr(self.columns) + "\n     #neurons: " + repr(self.rows)


neurons = 5
inputs = 2

hidden_layer = RNNLayer(number_of_neurons=neurons, external_inputs=inputs)
output_layer = Layer(1, neurons)

original_inputs = np.zeros(neurons)
inputs = np.asarray([1., 1.])

complete_input = np.append(original_inputs, inputs)

y = np.asarray([])

for i in xrange(50):

    if i < 50:
        inputs = np.zeros(2)
        #inputs = np.random.rand(2)
    elif i < 100:
        inputs = np.zeros(2)
    else:
        inputs = np.ones(2)

    #print "Complete inputs: ", complete_input

    h_o = hidden_layer.output(complete_input)
    complete_input = np.append(h_o, inputs)
    print "Hidden layer: ", h_o
    o = output_layer.output(h_o)
    y = np.append(y, h_o)

y = y.reshape((50, 5))

plt.plot(y[:, 0], 'b')
plt.plot(y[:, 1], 'r')
plt.plot(y[:, 2], 'g')
plt.plot(y[:, 3], 'y')
plt.plot(y[:, 4], 'k')

plt.show()

