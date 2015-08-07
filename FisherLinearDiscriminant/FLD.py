"""
Description: 
@author: Mingyuan Cui 
Created on: August 07 2015 11:54 AM
"""
import numpy as np
from numpy import linalg as LA


def importData():
    '''import Iris Data from bezdekIris.data.txt and return the list of
    three classes of input data'''

    rawData = np.genfromtxt('./data/bezdekIris.data', delimiter=",")
    a = np.zeros((50, 4))
    for i in range(0, 50):
        a[i] = rawData[i][0:4]
    b = np.zeros((50, 4))
    for i in range(50, 100):
        b[i - 50] = rawData[i][0:4]
    c = np.zeros((50, 4))
    for i in range(100, 150):
        c[i - 100] = rawData[i][0:4]
    a = np.matrix(a)
    b = np.matrix(b)
    c = np.matrix(c)
    a = a.transpose()  ##for the first clas
    b = b.transpose()  ##for the second clas
    c = c.transpose()  ##for the third clas
    d = [a, b, c]
    return d


def mean(input):
    '''calculate the mean of input data'''
    meanlist = np.mean(input, axis=1)
    meanlist.transpose()
    return meanlist


def sb(input1, input2, input3):
    '''calculate the SB matrix'''
    mean1 = mean(input1)
    mean2 = mean(input2)
    mean3 = mean(input3)
    meanall = (mean1 + mean2 + mean3) / 3
    sb = 50 * (mean1 - meanall) * ((mean1 - meanall).transpose()) + 50 * (mean2 - meanall) * (
        (mean2 - meanall).transpose()) + 50 * (mean3 - meanall) * ((mean3 - meanall).transpose())
    return sb


def sw(input1, input2, input3):
    '''calculate the SW matrix'''
    mean1 = mean(input1)
    mean2 = mean(input2)
    mean3 = mean(input3)
    a1 = np.matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  ##initial matrix of sk for the first class
    a2 = np.matrix(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  ##initial matrix of sk for the second class
    a3 = np.matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  ##initial matrix of sk for the third class
    for i in range(0, 50):
        b = input1[:, i]
        a1 = a1 + np.dot((b - mean1), (b - mean1).transpose())
    for i in range(0, 50):
        b = input2[:, i]
        a2 = a2 + np.dot((b - mean2), (b - mean2).transpose())
    for i in range(0, 50):
        b = input3[:, i]
        a3 = a3 + np.dot((b - mean3), (b - mean3).transpose())
    return a1 + a2 + a3


def eigen(sw, sb):
    '''calculate eigenvalues and eigenvectors of (SW^-1*SB)'''
    w, v = LA.eig(sw.getI() * sb)  ##w is the eigenvalues and v is the related eigenvectors
    return [w, v]


def eigenvecD2(w1, v):
    '''calculate the eigenvectors associated with the lagrest two eigenvalues'''

    w = w1
    ind = -1
    ind2 = -1
    maxh = max(w)
    vout = []  ##initial the pair of two eigenvalues for 2-D
    for i in range(0, len(w)):
        if w[i] == maxh:
            ind = i
            vout.append(w[i])
            w[i] = -9999999  ##rewrite the largest eigenvalue to find the second largest eigenvalue
    maxh2 = max(w)
    for i in range(0, len(w)):
        if w[i] == maxh2:  ##the second largest eigenvalue
            ind2 = i
            vout.append(w[i])
    newMatrix = np.concatenate((v[ind], v[ind2]), axis=0)  ##the estimated matrix of W
    return [vout, newMatrix.transpose()]


def inputD2(input2, w):
    '''calucalate the projected 2-D input'''

    return np.dot(w, input2)


def plotClasses(inputD21, inputD22, inputD23):
    '''plot the projected 2-D data in three classes'''
    import matplotlib.pyplot as plt

    plt.axis([-10, 10, -10, 10])
    plt.plot(inputD21[0], inputD21[1], 'ro', inputD22[0], inputD22[1], 'bo', inputD23[0], inputD23[1], 'go')
    plt.show()


def main():
    '''main function to calculate my Jw'''


def ratio(wt, sw, sb):
    '''calculate the ratio j using w Sw Sb'''
    w = wt.transpose()

    a = np.dot(np.dot(w, sw), w.transpose())  ##W S_w W^T
    b = np.dot(np.dot(w, sb), w.transpose())  ##W S_b W^T
    c = np.trace(a.getI() * b)
    return c


def J(inputD21, inputD22, inputD23):
    '''2.1.3 an alternative way to calculate the ratio j using sw sb'''

    inputAll = np.concatenate((inputD21, inputD22), axis=1)
    inputAll = np.concatenate((inputAll, inputD23), axis=1)
    miuAll = mean(inputAll)
    miu1 = mean(inputD21)
    miu2 = mean(inputD22)
    miu3 = mean(inputD23)
    a1 = np.matrix([[0, 0], [0, 0]])
    a2 = np.matrix([[0, 0], [0, 0]])
    a3 = np.matrix([[0, 0], [0, 0]])
    for i in range(0, 50):
        b = inputD21[:, i] - miu1
        a1 = a1 + np.dot(b, b.transpose())
    for i in range(0, 50):
        b = inputD22[:, i] - miu2
        a2 = a2 + np.dot(b, b.transpose())
    for i in range(0, 50):
        b = inputD23[:, i] - miu3
        a3 = a3 + np.dot(b, b.transpose())
    sw_p = a1 + a2 + a3
    sb_p = 50 * np.dot(miu1 - miuAll, (miu1 - miuAll).transpose()) + 50 * np.dot(miu2 - miuAll, (
        miu2 - miuAll).transpose()) + 50 * np.dot(miu3 - miuAll, (miu3 - miuAll).transpose())
    j = np.trace(np.dot(sw_p.getI(), sb_p))
    return j


if __name__ == "__main__":
    l = importData()
    sw1 = sw(l[0], l[1], l[2])
    sb1 = sb(l[0], l[1], l[2])

    w = eigen(sw1, sb1)
    e = eigenvecD2(w[0], w[1].transpose())
    r = ratio(e[1], sw1, sb1)
    inputD21 = inputD2(l[0], e[1].transpose())
    inputD22 = inputD2(l[1], e[1].transpose())
    inputD23 = inputD2(l[2], e[1].transpose())
    j = J(inputD21, inputD22, inputD23)
    plotClasses(inputD21, inputD22, inputD23)
    print j
