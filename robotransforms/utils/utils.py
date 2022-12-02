import numpy as np

import utils_wrapper as wrap 

def cholesky(v):
    out = np.zeros((len(v),len(v)))
    wrap.cholesky(v, out)
    return out

def get_sigma_points(x,cov):
    n = len(x)
    X = np.zeros((2*n+1, n))
    wrap.get_sigma_points(n,x,cov,X)
    return X

def GRV_statistics(X,L):
    n = len(X[0])
    x, cov = np.zeros(n), np.zeros((n,n))
    wrap.GRV_statistics(n,L,X,x,cov)
    return x, cov

def UT(x, cov, func):
    Y = np.vectorize(func, signature="(n)->(n)")(get_sigma_points(x, cov))
    return GRV_statistics(Y,len(x))

