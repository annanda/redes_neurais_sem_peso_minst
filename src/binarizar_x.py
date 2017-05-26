# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_mldata
import csv


def binarizar_x(xd, threshold):
    temp = list(xd)
    result = temp[:]
    for i in xrange(len(result)):
        for j in xrange(len(result[i])):
            if result[i][j] >= threshold:
                result[i][j] = 1
            else:
                result[i][j] = 0
    return result


def create_file(file_name, content):
    with open('../binary_x/th_' + file_name + '.csv', 'wb') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=',')
        for line in content:
            spamwriter.writerow(line)


def main():

    for i in xrange(2, 5):
        mnist = fetch_mldata('MNIST original')
        X = mnist.data
        threshold = i
        result = binarizar_x(X, threshold)
        create_file(str(threshold), result)
        print "Criado dataset do th {}".format(threshold)


if __name__ == '__main__':
    main()


