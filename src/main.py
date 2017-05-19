# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from PyWANN.WiSARD import WiSARD

mnist = fetch_mldata('MNIST original')

y = mnist.target
X = mnist.data

'''
Os dados em y vêm de 0. a 9.
É preciso transformá-los para notação binária para ser usada pela PyWANN
'''

'''
Retorna uma lista cujos índices são representações binárias dos valores de y

Por exemplo, 
    0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
'''


def formatar_y():
    y_formated = []
    for i in xrange(len(y)):
        y_i = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_i[int(y[i])] = 1
        y_formated.append(y_i)
    return y_formated


yBin = y  # formatarY(y)

y_train, y_test = yBin[:60000], yBin[60000:]


def binarizar_x(xd, threshold):
    result = list(xd)
    for i in xrange(len(result)):
        for j in xrange(len(result[i])):
            if result[i][j] >= threshold:
                result[i][j] = 1
            else:
                result[i][j] = 0
    return result

xb = binarizar_x(X, 1)
print "Terminado"


y_t = []  # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]

k = 0
p = 0

X_train = xb[:60000]

x_t = []

maxe = 50

for i in xrange(len(X_train)):
    if (int(y_train[i]) == k):
        p += 1
        x_t.append(X_train[i])
        y_t.append(k)
    if (p == maxe):
        p = 0
        k += 1
    if k == maxe and p == maxe:
        break

num_bits_addr = 7
randomize_positions = False
bleaching = False

w = WiSARD(num_bits_addr, bleaching, randomize_positions)

w.fit(x_t, y_t)

predicted = w.predict(xb[60000:])
expected = y_test

accuracy = accuracy_score(predicted, expected)
print ("\nAccuracy: %s" % accuracy)