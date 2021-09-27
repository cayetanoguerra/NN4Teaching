import numpy as np

x_data = [[0., 0.],
          [1., 0.],
          [0., 1.],
          [1., 1.]]

y_data = [1, 1, 0, 0]


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_derivate(o):
    return o * (1.0 - o)


def train(x_data, y_data):

    w0, w1, w2 = (np.random.rand(3)*2. - 1.)
    lr = 0.001
    epochs = 1000

    print("Training...")

    for _ in range(epochs):
        
        w0_d = []
        w1_d = []
        w2_d = []
        
        for data, label in zip(x_data, y_data):

            o = sigmoid(w0*1.0 + w1*data[0] + w2*data[1])
            temp = 2.*(o - label) * sigmoid_derivate(o)

            w0_d.append(temp * 1.0)      # e_0 = 1
            w1_d.append(temp * data[0])  # e_1 = data[0]
            w2_d.append(temp * data[1])  # e_2 = data[1]
            
        w0 = w0 - np.sum(w0_d) * lr
        w1 = w1 - np.sum(w1_d) * lr
        w2 = w2 - np.sum(w2_d) * lr
        
        
    # Testing...

    for data, label in zip(x_data, y_data):
        print(data, "->", label)
        o = sigmoid(w0*1.0 + w1*data[0] + w2*data[1])
        print(o)
        print("-----------------------")

    e_1 = 0.0
    e_2 = 0.5
    print(sigmoid(w0*1.0 + w1*e_1 + w2*e_2))


train(x_data, y_data)
