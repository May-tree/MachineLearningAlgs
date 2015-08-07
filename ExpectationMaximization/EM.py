"""
Description: 
@author: Mingyuan Cui 
Created on: August 07 2015 12:01 PM
"""

import numpy as np
import math
from scipy import random, linalg


def importData():
    # import Iris Data from bezdekIris.data.txt'''
    rawData = np.genfromtxt('./data/bezdekIris.data.txt', delimiter=",")
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


def pdf(mean1, coV1, x):
    # to estimate the probablity that the newinput is belong to this class
    n = len(mean1)
    inv = np.linalg.inv(coV1)
    det = (np.linalg.det(coV1))
    q1 = 1.0 / ((2 * math.pi) ** (0.5 * n) * (det ** 0.5))
    q2 = math.exp(-0.5 * (x - mean1).transpose() * inv * (x - mean1))
    q = q1 * q2
    return q


def initialize(input):
    # to initialize the MLE parameters
    a = random.uniform(0, 150)
    b = random.uniform(0, 150)
    c = random.uniform(0, 150)
    miu = [input[:, a], input[:, b],
           input[:, c]]  # set the initial mu to be the same as randomly chosen input from the original inputs
    cov = [np.matrix(np.eye(4)), np.matrix(np.eye(4)),
           np.matrix(np.eye(4))]  # set the initial covariances to be identities
    a = random.random()
    b = random.random()
    c = random.random()
    a1 = a / a + b + c
    b1 = b / a + b + c
    c1 = c / a + b + c
    pai = [a1, b1, c1]  # set the initial pi to be three normalized random figure, and sums up to 1
    return [pai, miu, cov]


def llh(pailist, miulist, coVlist, input):
    # to estimate the log-likelihood
    sumn = 0
    for i in range(0, input.shape[1]):
        sumk = 0
        for j in range(0, 3):
            sumk += pailist[j] * pdf(miulist[j], coVlist[j], input[:, i])
        sumn += np.log(sumk)
    return sumn


def ellh(pailist, miulist, coVlist, input, gammatrix):
    # to estimate the expected log-likelihood
    sumn = 0
    for i in range(0, input.shape[1]):
        sumk = 0
        for j in range(0, 3):
            sumk += gammatrix[j, i] * np.log(pailist[j] * pdf(miulist[j], coVlist[j], input[:, i]))
        sumn += sumk
    return sumn


def E_step(pailist, miulist, coVlist, input):
    # E-STEP to get the gamma matrix
    gammatrix = np.matrix(np.zeros((3, input.shape[1])))
    for i in range(0, input.shape[1]):
        for j in range(0, 3):
            down = 0
            for k in range(0, 3):
                down += pailist[k] * pdf(miulist[k], coVlist[k], input[:, i])
            gammatrix[j, i] = pailist[j] * pdf(miulist[j], coVlist[j], input[:, i]) / down
    return gammatrix


def M_step(gammatrix, input):
    # M-STEP to generate new parameters
    Nk = [0, 0, 0]
    miulist = [np.matrix(np.zeros((4, 1))), np.matrix(np.zeros((4, 1))), np.matrix(np.zeros((4, 1)))]
    for i in range(0, 3):
        for j in range(0, input.shape[1]):
            Nk[i] += gammatrix[i, j]
            miulist[i] += gammatrix[i, j] * input[:, j]
        miulist[i] /= Nk[i]
    coVlist = [np.matrix(np.zeros((4, 4))), np.matrix(np.zeros((4, 4))), np.matrix(np.zeros((4, 4)))]
    for i in range(0, 3):
        for j in range(0, input.shape[1]):
            coVlist[i] += gammatrix[i, j] * (input[:, j] - miulist[i]) * ((input[:, j] - miulist[i]).transpose())
        coVlist[i] /= Nk[i]
    pailist = [0, 0, 0]
    for i in range(0, 3):
        pailist[i] += Nk[i] / input.shape[1]
    return [pailist, miulist, coVlist]


def singularCheck(coVlist):
    # sometimes it generates singular covariance, so it is needed that we check this and re-initialzied he means as well as covariances
    for coV in coVlist:
        if np.linalg.det(coV) == 0:
            return True
    return False


if __name__ == '__main__':
    input = importData()[0:4, :]
    init = initialize(input)
    pailist = init[0]
    miulist = init[1]
    coVlist = init[2]
    hood = llh(pailist, miulist, coVlist, input)  # the initial log-likelihood
    thresh = 0.001  # set the threshold for the difference between two continous log-likelihood
    iter = 0
    gammatrix = np.matrix(np.zeros((3, 150)))
    while iter < 100:  # set the maximum of the number of iterations to be 100
        gammatrix = E_step(pailist, miulist, coVlist, input)
        newpara = M_step(gammatrix, input)
        pailist = newpara[0]
        miulist = newpara[1]
        coVlist = newpara[2]
        ehood = ellh(pailist, miulist, coVlist, input, gammatrix)
        if singularCheck(coVlist) == True:
            miulist = initialize(input)[1]
            coVlist = initialize(input)[2]
            print "singular cov find"
            continue
        newhood = llh(pailist, miulist, coVlist, input)
        differ = newhood - hood
        print round(ehood, 3), "&", round(newhood, 3), "&", iter + 1
        if abs(differ) < thresh:
            break
        hood = newhood
        iter += 1
    print "converged"
