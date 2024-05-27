# 9.)Implement the non-parametric Locally Weighted Regressionalgorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def kernel(point,xmat,k):
    m,n = np.shape(xmat) # count of values
    weights = np.mat(np.eye(m)) # identity matrix
    for j in range(m):
        diff = point - xmat[j] # diff in currvalue - jth value
        # change value of jth diagonal element
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights

def localweight(point,xmat,ymat,k):
    #form kernel
    wei = kernel(point,xmat,k)
    # calculate weight
    W = (xmat.T * (wei * xmat)).I * (xmat.T * (wei*ymat.T))
    return W

def localweightRegression(xmat,ymat,k):
    m,n = np.shape(xmat) # count of values
    ypred = np.zeros(m) # pred array = [0,0,...]
    for i in range(m):
        # pred = xmatValue * localWeight
        ypred[i] = xmat[i] * localweight(xmat[i],xmat,ymat,k)
    return ypred

def GraphPlot(X,ypred):
    sortindex = X[:,1].argsort(0)
    xsort = X[sortindex][:,0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(bill,tip,color = 'green')
    ax.plot(xsort[:,1],ypred[sortindex],color = 'red',linewidth = 5)
    plt.xlabel('Total Bill')
    plt.ylabel('Tip')
    plt.show()

data = pd.read_csv('./p9.csv')
bill = np.array(data.total_bill) # bill array
tip = np.array(data.tip) # tip array
mbill = np.mat(bill) # bill matrix
mtip = np.mat(tip)  # tip matrix
m = np.shape(mbill)[1]  # count of values
one = np.mat(np.ones(m))  # matrix with 1s
X = np.hstack((one.T,mbill.T))  # 2d array of 1, bill

ypred = localweightRegression(X,mtip,3)
GraphPlot(X,ypred)
