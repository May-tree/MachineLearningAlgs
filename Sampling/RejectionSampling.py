"""
Description: 
@author: Mingyuan Cui 
Created on: August 07 2015 12:00 PM
"""

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import math
import time


def wallaby(x):
    # pdf of wallaby distribution
    return 0.4 * norm.pdf(x, 2, 20) + 0.3 * norm.pdf(x, 9, 2) + 0.3 * norm.pdf(x, 5, 0.5)


def trans(y, loc, scale):
    # convert the uniformly distributed figure to be the sample from cauchy distribution
    return scale * np.tan(y) + loc


def r_sample(num, loc, scale, k):
    # to do the rejection sampling for k times regardless of whether it is rejected
    u2 = np.random.uniform(-0.5 * math.pi, 0.5 * math.pi,
                           num)  # to generate a uniformly distributed number from -1/2*pi to 1/2*pi
    x = trans(u2, loc, scale)  # to generate a number cauchy distribution
    var = cauchi(x, loc, scale, k)  # to calculate the pdf of cauchy distribution muliplied by k
    tar = wallaby(x)  # to calculate the pdf of wallaby distribution
    y = np.random.uniform(np.zeros(num),
                          var)  # to generate uniformly distributed numbers between 0 to the pdf of cauchy distributuion
    sample = x[(y < tar).nonzero()]  # to filter out the rejected samples
    if type(y) == float:  # this is to avoid when the array of y has only element, it will be considered as a float.
        y = np.array([y])
    sampley = y[(y < tar).nonzero()]
    reject = num - len(sample)
    return sample, sampley, reject


def getSample(loc, scale, k):
    sample, sampley, reject = r_sample(1000000, loc, scale, k)
    while len(sample) < 1000000:  # this loop is to gain the samples by the number of how many samples still needs
        rest = 1000000 - len(sample)
        sample1, sampley1, reject1 = r_sample(rest, loc, scale, k)
        sample = np.concatenate((sample, sample1), axis=1)
        sampley = np.concatenate((sampley, sampley1), axis=1)
        reject += reject1
    return sample, sampley, reject


def test(sample):
    # to calculated the sum of the squared errors
    tests = np.arange(-49.95, 50.05, 0.1)  # make bins
    true = 0.4 * norm.pdf(tests, 2, 20) + 0.3 * norm.pdf(tests, 9, 2) + 0.3 * norm.pdf(tests, 5,
                                                                                       0.5)  # the true pdf of wallaby distribution
    predicate = np.zeros(1000)
    for i in range(0, 1000):  # sum the errors up
        k1 = sample[(sample <= -49.95 + (i + 1) * 0.1).nonzero()]
        k2 = k1[(k1 >= -49.95 + i * 0.1).nonzero()]
        k = len(k2)
        predicate[i] = k / (1000000 * 0.1)
    differ = true - predicate
    return sum(differ * differ)


def cauchi(x, loc, scale, k):
    return k / ((1 + ((x - loc) / scale) ** 2))


if __name__ == '__main__':
    start_time = time.time()
    loc = 5
    scale = 3
    k = 0.2528
    sample, sampley, reject = getSample(loc, scale, k)
    print "rejections:", reject
    print "runtime:", time.time() - start_time
    y = 0.4 * norm.pdf(sample[10000:20000], 2, 20) + 0.3 * norm.pdf(sample[10000:20000], 9, 2) + 0.3 * norm.pdf(
        sample[10000:20000], 5, 0.5)
    plt.plot(sample[10000:20000], sampley[10000:20000], "ro")
    plt.plot(sample[10000:20000], y, "bo")
    plt.axis([-20, 20, 0, 0.3])
    plt.show()
    print "sum of the squared errors", test(sample)
