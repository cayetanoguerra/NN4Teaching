import numpy as np

x_data = [[0, 0],
          [10, 0],
          [0, 10],
          [10, 10]]

y_data = [1, 1, 0, 0]


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_derivate(o):
    return o * (1.0 - o)


def train(x_data, y_data):

    w1, w2, b = np.random.rand(3)

    lr = 0.1

    print "Training..."

    for j in range(10000):
        
        w1_d = []
        w2_d = []
        b_d = []
        for data, label in zip(x_data, y_data):

            o = sigmoid(w1*data[0] + w2*data[1] + b)

            error = 2.*(o - label) * sigmoid_derivate(o)

            w1_d.append(error * data[0])
            w2_d.append(error * data[1])
            b_d.append(error)

        w1 = w1 - (np.sum(w1_d)/4.) * lr
        w2 = w2 - (np.sum(w2_d)/4.) * lr
        b = b - (np.sum(b_d)/4.) * lr

        

    for data, label in zip(x_data, y_data):
        print data, "->", label
        o = sigmoid(w1*data[0] + w2*data[1] + b)
        print o
        print "-----------------------"


train(x_data, y_data)



