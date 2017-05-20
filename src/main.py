# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from PyWANN.WiSARD import WiSARD
import csv


def read_y():
    mnist = fetch_mldata('MNIST original')
    y = mnist.target
    y_train, y_test = y[:60000], y[60000:]
    return y_train,y_test


def read_x(threshold):
    x = []
    with open('../binary_x/' + str(threshold) +'.csv', 'rb') as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            line = line.split(',')
            line = map(int, line)
            x.append(line)
    return x


def get_n_examples_each_class(n_examples, x_train, y_train):
    x_t = []
    y_t = []
    contador_exemplos = 0
    digito = 0
    for i in xrange(len(x_train)):
        if (int(y_train[i]) == digito):
            contador_exemplos += 1
            x_t.append(x_train[i])
            y_t.append(digito)
        if (contador_exemplos == n_examples):
            contador_exemplos = 0
            digito += 1
        if digito == n_examples and contador_exemplos == n_examples:
            break
    return x_t, y_t


def apply_wisard(x_train, y_train, x_test, y_test, num_bits_addr, randomize_positions, bleaching):
    w = WiSARD(num_bits_addr, bleaching, randomize_positions)
    w.fit(x_train, y_train)
    predicted = w.predict(x_test)
    expected = y_test
    accuracy = accuracy_score(predicted, expected)
    return accuracy


def test_thresholds():
        for i in xrange(67, 71):
            x_threshold = read_x(i)
            x_train, x_test = x_threshold[:60000], x_threshold[60000:]
            y_train, y_test = read_y()
            num_bits_addr = 32
            randomize_positions = True
            bleaching = True
            accuracy = apply_wisard(x_train, y_train, x_test, y_test, num_bits_addr, randomize_positions, bleaching)
            with open('results_' + str(i) + '.csv', 'w') as csv_file:
                spamwriter = csv.writer(csv_file, delimiter=',')
                spamwriter.writerow([i, accuracy])
            print 'Feito o threshold {}'.format(i)

if __name__ == '__main__':
    test_thresholds()
