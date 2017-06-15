# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import KFold
from PyWANN.WiSARD import WiSARD
import csv
import click

QUANTIDADE_EXEMPLOS = 60000


def read_y():
    mnist = fetch_mldata('MNIST original')
    y = mnist.target
    # y_train, y_test = y[:QUANTIDADE_EXEMPLOS], y[60000:]
    # return y_train, y_test
    return y


def read_x(threshold):
    x = []
    with open('../binary_x/th_' + str(threshold) +'.csv', 'rb') as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            line = line.split(',')
            line = map(int, line)
            x.append(line)
    return x


def generate_folds(X, y):
    x_train_folds = []
    x_test_folds = []
    y_train_folds = []
    y_test_folds = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    kf = KFold(n_splits=4)

    for train_index, test_index in kf.split(X):
        for idx in train_index:
            x_train.append(X[idx])
            y_train.append(y[idx])
        for idx_y in test_index:
            x_test.append(X[idx_y])
            y_test.append(y[idx_y])
        x_train_folds.append(x_train)
        x_test_folds.append(x_test)
        y_train_folds.append(y_train)
        y_test_folds.append(y_test)
        print 'tamanho x train: {}'.format(len(x_train))
        print 'tamanho x teste: {}'.format(len(x_test))
        x_train = []
        x_test = []
        y_train = []
        y_test = []
    print 'tamanho x fold: {}'.format(len(x_train_folds))
    print x_train_folds[1][1]
    return x_train_folds, y_train_folds, x_test_folds, y_test_folds


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


def apply_wisard(x_train_folds, y_train_folds, x_test_folds, y_test_folds, num_bits_addr, randomize_positions, bleaching):
    w = None
    accuracies = []
    for fold_number in xrange(4):
        w = WiSARD(num_bits_addr, bleaching, randomize_positions)
        w.fit(x_train_folds[fold_number], y_train_folds[fold_number])
        predicted = w.predict(x_test_folds[fold_number])
        expected = y_test_folds[fold_number]
        accuracy = accuracy_score(predicted, expected)
        print 'Accuracy {}: {}'.format(fold_number, accuracy)
        accuracies.append(accuracy)
    return accuracies

@click.command()
@click.option('--num_bits_addr', type=click.INT, default=32, help='Numero de addr memory')
@click.option('--randomize_positions', type=click.BOOL, default=True, help='Randomize position')
@click.option('--bleaching', type=click.BOOL, default=True, help='Bleaching')
@click.option('--threshold', type=click.INT, default=1, help='Threshold do dataset')
def test_thresholds(num_bits_addr, randomize_positions, bleaching, threshold):
        i = threshold
        x = read_x(i)
        y = read_y()
        x_train_folds, y_train_folds, x_test_folds, y_test_folds = generate_folds(x, y)
        num_bits_addr = num_bits_addr
        randomize_positions = randomize_positions
        bleaching = bleaching
        accuracies = apply_wisard(
            x_train_folds,
            y_train_folds,
            x_test_folds,
            y_test_folds,
            num_bits_addr,
            randomize_positions,
            bleaching
        )
        # print(accuracies)
        print '{},{},{}'.format(i, np.mean(accuracies), np.std(accuracies))
        # with open('results_threshold_' + str(i) + '.csv', 'w') as csv_file:
        #     spamwriter = csv.writer(csv_file, delimiter=',')
        #     spamwriter.writerow([i, accuracy])
        #     # print 'threshod {}, acuracia {}'.format(i, accuracy)
        #     print '{},{}'.format(i, accuracy)
        # print 'Feito o threshold {}'.format(i)


if __name__ == '__main__':
    test_thresholds()
