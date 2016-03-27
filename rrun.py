import numpy as np
import rnn4t

import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi * 50, 5000)

x1 = np.cos(x)*0.1 + 0.5
x2 = np.sin(x)*0.1 + 0.5


x_data = []
y_data = []

for i in xrange(4900):
    x_data.append(zip(x1[i:i+10], x2[i:i+10]))
    y_data.append((x1[i+10], x2[i+10]))

# print y_data
#
# print "\nSome samples..."
# for i in range(20):
#     print x_data[i], " -> ", y_data[i]
# print
# x_data = [[-1., 1.],
#           [-2., 2.],
#           [-3., 3.],
#           [-4., 4.],
#           [50., 5.],
#           [60., 6.],
#           [70., 7.]]
#
# y_data = [1., 0., 0., 0., 1., 1., 1.]
# x_data = [[0, 0],
#           [1, 0],
#           [0, 1],
#           [1, 1]]
#
# y_data = [1, 0, 0, 1]
# y_data = [1., 1., 1., 1., 0., 0., 0.]


rnet = rnn4t.RNet(layers=[2, 10, 2])

error_plot = []

for i in range(450):
    print "Epoch: ", i
    e = rnet.train(x_data, y_data, lr=0.1)
    print "Error: ", e
    error_plot.append(e)

plots_x = []
plots_y = []

for i in range(500):
    x_data = zip(x1[i:i+10], x2[i:i+10])
    output = rnet.output(x_data)
    plots_x.append(output[0])
    plots_y.append(output[1])


#plt.plot(plots_x, plots_y, 'bs', x1, x2, 'rs')
plt.plot(plots_x, plots_y, 'bs')
plt.show()

plt.plot(error_plot)
plt.show()

x_data = zip(x1[0:0+10], x2[0:0+10])

plots_x = []
plots_y = []
for i in range(500):
    output = rnet.output(x_data)
    plots_x.append(output[0])
    plots_y.append(output[1])
    x_data.pop(0)
    x_data.append(output)

plt.plot(plots_x, plots_y, 'rs')
plt.show()


#
# for i in range(10000):
#     print "Epoch: ", i
#     print net.train(x_data, y_data)
#
# print net.layers[0]
#
# for i in range(len(x_data)):
#     print y_data[i]
#     print net.output(x_data[i])


