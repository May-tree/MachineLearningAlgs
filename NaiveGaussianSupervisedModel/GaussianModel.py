"""
Description: 
@author: Mingyuan Cui 
Created on: August 07 2015 12:08 PM
"""

import numpy as np
from scipy.stats import multivariate_normal


def importData():
    # import input data
    rawData = np.genfromtxt('./data/bezdekIris.data', delimiter=",")
    a = np.zeros((150, 5))
    for i in range(0, 50):
        a[i] = rawData[i]
        a[i][4] = 0
    for i in range(50, 100):
        a[i] = rawData[i]
        a[i][4] = 1
    for i in range(100, 150):
        a[i] = rawData[i]
        a[i][4] = 2
    b = np.matrix(a.transpose())
    return b


def tenfold(input):
    # ten-fold cv
    np.random.shuffle(np.transpose(input))  # randomlize the inputs
    errorlist = []
    for i in range(0, 10):
        test = input[0:4, i * 15:(i + 1) * 15]  # test data
        truth = input[4, i * 15:(i + 1) * 15]  # the ground truth of classes
        train = deleteCol(input, i * 15, (i + 1) * 15)  # training data
        indicate = [0, 0, 0]  # trivial indicaters to judge if a distinct class is found
        for j in range(0, train.shape[1]):  # this loop is to split the train data into three classes
            if train[4, j] == 0:
                if indicate[0] == 0:
                    train1 = train[0:4, j]
                    indicate[0] = 1
                else:
                    train1 = np.concatenate((train1, train[0:4, j]), axis=1)
            elif train[4, j] == 1:
                if indicate[1] == 0:
                    train2 = np.matrix(train[0:4, j])
                    indicate[1] = 1
                else:
                    train2 = np.concatenate((train2, train[0:4, j]), axis=1)
            else:
                if indicate[2] == 0:
                    train3 = np.matrix(train[0:4, j])
                    indicate[2] = 1
                else:
                    train3 = np.concatenate((train3, train[0:4, j]), axis=1)
        meanlist = [np.mean(train1, axis=1), np.mean(train2, axis=1),
                    np.mean(train3, axis=1)]  # to compute the means of three classes
        coVlist = [np.matrix(np.cov(train1)), np.matrix(np.cov(train2)),
                   np.matrix(np.cov(train3))]  # to compute the covariances of three classes
        gaus = [prob(meanlist[0], coVlist[0], test), prob(meanlist[1], coVlist[1], test),
                prob(meanlist[2], coVlist[2], test)]  # generate the three gaussian predictions for tests data
        error = 0
        for j in range(0, test.shape[1]):
            max = 0;
            probi = 0;
            for g in range(0, 3):  # to estimate the max probablity
                if gaus[g][j] > probi:
                    probi = gaus[g][j]
                    max = g
            if max != truth[0, j]:  # validation
                error += 1
        errorlist.append(error)
    return (float)(sum(errorlist)) / len(errorlist)


def deleteCol(matrice, i, j):
    # to delete the test sets from the original sets, thus makes the rest to be training set
    a = matrice[:, 0:i]
    b = matrice[:, j:]
    c = np.concatenate((a, b), axis=1)
    return c


def prob(mean1, coV1, input):
    # to estimate the probablity that the newinput is belong to this class
    mean = np.squeeze(np.asarray(mean1))
    coV = np.squeeze(np.asarray(coV1))
    var = multivariate_normal(mean, coV)
    p = var.pdf(np.squeeze(np.asarray(input.transpose())))
    return p


if __name__ == '__main__':
    # main function to calculate error
    d = importData()
    print tenfold(d)
