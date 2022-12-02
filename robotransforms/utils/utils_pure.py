import numpy as np

L_PLUS_LAMBDA = 3

def cholesky(v):
    return np.linalg.cholesky(v).T

def get_sigma_points(x,cov):
    L = len(x)
    LAMBDA = L_PLUS_LAMBDA - L

    X = np.zeros((2*L+1,L))
    X[0] = x
    scaled_cov = cov * L_PLUS_LAMBDA
    # Rows of the R matrix in A=R^TR function like a square root
    dx = cholesky( scaled_cov )
    for i in range(L):
        X[1 + i] = x + dx[i]
        X[1 + i + L] = x - dx[i]

    return X

def GRV_statistics(X,L):
    M = len(X[0])
    LAMBDA = L_PLUS_LAMBDA - L
    W0 = LAMBDA / L_PLUS_LAMBDA
    W = 1 / ( 2 * L_PLUS_LAMBDA )

    x_bar = X[0]*W0 + np.sum(X[1:]*W,axis=0)

    cov = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            cov[i,j] = W0 * (X[0,i] - x_bar[i]) * (X[0,j] - x_bar[j])
            for k in range(1,len(X)):
                cov[i,j] = cov[i,j] + W * (X[k,i] - x_bar[i]) * (X[k,j] - x_bar[j])

    return x_bar, cov

def UT(x, cov, func):
    Y = np.vectorize(func, signature="(n)->(n)")(get_sigma_points(x, cov))
    return GRV_statistics(Y,len(x))
