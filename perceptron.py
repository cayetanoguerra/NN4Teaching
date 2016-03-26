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

    for j in range(10000):
        print "Training..."
        w1_d = []
        w2_d = []
        b_d = []
        for i in range(len(x_data)):

            o = sigmoid(w1*x_data[i][0] + w2*x_data[i][1] + b)

            error = 2.*(o - y_data[i]) * sigmoid_derivate(o)

            w1_d.append(error * x_data[i][0])
            w2_d.append(error * x_data[i][1])
            b_d.append(error)

        w1 = w1 - (np.sum(w1_d)/4.) * lr
        w2 = w2 - (np.sum(w2_d)/4.) * lr
        b = b - (np.sum(b_d)/4.) * lr

        print w1, w2, b

    for i in range(len(x_data)):
        o = sigmoid(w1*x_data[i][0] + w2*x_data[i][1] + b)
        print o

    print w1, w2, b

train(x_data, y_data)



