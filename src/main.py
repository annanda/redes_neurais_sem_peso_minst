# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from PyWANN.WiSARD import WiSARD

mnist = fetch_mldata('MNIST original')

y = mnist.target


def read_x():
    x = []
    with open('../binary_x/1.csv', 'rb') as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            line = line.split(',')
            line = map(int, line)
            x.append(line)
    return x

x_threshold = read_x()
x_train, x_test = x_threshold[:60000], x_threshold[60000:]
y_train, y_test = y[:60000], y[60000:]
y_t = []  # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]

k = 0
p = 0

x_t = []

maxe = 50

for i in xrange(len(x_train)):
    if (int(y_train[i]) == k):
        p += 1
        x_t.append(x_train[i])
        y_t.append(k)
    if (p == maxe):
        p = 0
        k += 1
    if k == maxe and p == maxe:
        break
#
num_bits_addr = 7
randomize_positions = False
bleaching = False

w = WiSARD(num_bits_addr, bleaching, randomize_positions)

w.fit(x_t, y_t)

predicted = w.predict(x_test)
expected = y_test

accuracy = accuracy_score(predicted, expected)
print ("\nAccuracy: %s" % accuracy)

# if __name__ == '__main__':
#     x = read_x()
#     print x
