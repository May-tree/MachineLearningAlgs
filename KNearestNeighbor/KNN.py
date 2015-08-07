"""
Description: 
@author: Mingyuan Cui 
Created on: August 07 2015 12:19 PM
"""
import math
import numpy as np

def kNN(k,train1,test1):
    '''calculate the errors of a k-NN method'''
    train=train1[0:train1.shape[0]-1] ##n-dimentional training set
    trainy=train1[train1.shape[0]-1] ##ground-truth of label for training set
    test=test1[0:test1.shape[0]-1] ##n-dimentional test set
    testy=test1[test1.shape[0]-1] ##ground-truth of label for test set
    n=train.shape[0] ##dimention
    m=train.shape[1] ##number of training data
    m2=test.shape[1] ##number of test data
    testyc=np.matrix(np.zeros(m2))
    for i in range(0,m2):
        em=np.zeros(m) ##initial Euclidean distance matrix
        for l in range(0,m):
            d2=0
            for j in range(0,n):
                d2=d2+(test[j,i:i+1]-train[j,l:l+1])*(test[j,i:i+1]-train[j,l:l+1])
            d=math.sqrt(d2) ## Euclidean distance
            em[l]=d

        acent=np.argsort(em)[0:k] ## re-order and get the first k training data which has the shortest distances

        scorelist=np.array(np.zeros(k)) ## initial label frequency array
        for i2 in range(0,k):
            scorelist[i2]=trainy[:,acent[i2]:acent[i2]+1]
        count=np.bincount(scorelist.astype(np.int32)) ## the biggest frequeny
        testyc[:i:i+1]=np.argmax(count) ## the most frequent class amongst its k nearest neighbors, that is abtained from the biggest frequency
    error =0
    for i in range(0,m2):
        if testyc[:,i:i+1] != testy[:,i:i+1]:
            error=error+1 ##accumulate the classifying error
    return error

def sfold(k,s,rawData):
    '''calculate the average error of s-fold methold'''
    N=rawData[0].shape[1] ##the number of all data
    g=N/s ##the size of each group in s-fold method
    error=0
    for i in range(0,s):
        test=rawData[:,i*g:(i+1)*g]
        train=np.concatenate((rawData[:,0:i*g], rawData[:,(i+1)*g:N]), axis=1)
        error=error+kNN(k,train,test) ## accumulated error plus the error for the ith test
    return float(error)/s ## calculate the average error

def importData():
    '''import input data'''
    rawData=np.genfromtxt('./data/bezdekIris.data.txt', delimiter=",")
    a=np.zeros((150,5))
    for i in range(0,50):
        a[i]=rawData[i]
        a[i][4]=0
    for i in range(50,100):
        a[i]=rawData[i]
        a[i][4]=1
    for i in range(100,150):
        a[i]=rawData[i]
        a[i][4]=2
    a=a.transpose()
    b=a[4]
    d=[np.matrix(a[0:4]),np.matrix(b)]
    return d

if __name__=='__main__':
    k=3
    s=10
    origin=importData()
    sfold(k,s,origin)



