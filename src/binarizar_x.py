# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_mldata
import csv

mnist = fetch_mldata('MNIST original')

X = mnist.data


def binarizar_x(xd, threshold):
    result = list(xd)
    for i in xrange(len(result)):
        for j in xrange(len(result[i])):
            if result[i][j] >= threshold:
                result[i][j] = 1
            else:
                result[i][j] = 0
    return result


def create_file(file_name, content):
    with open('../binary_x/' + file_name + '.csv', 'wb') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=',')
        for line in content:
            spamwriter.writerow(line)


def main():
    threshold = 1
    result = binarizar_x(X, threshold)
    create_file(str(threshold), result)


if __name__ == '__main__':
    main()


