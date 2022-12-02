#include <math.h>

#include "base.h"

// Adapted from http://www.seas.ucla.edu/~vandenbe/133A/lectures/chol.pdf
int cholesky_step( double *R, int n, int i ) {
    int _i,_j;
    // A[i:n][i:n] - R[i-1,i:n]^T R[i-1,i:n]
    if ( i != 0.0 ) {
        for ( _i = i; _i < n; _i ++ ) {
            R[_i*n + _i] -= R[(i-1)*n + _i] * R[(i-1)*n + _i];
            for ( _j = _i+1; _j < n; _j ++ ) {
                double temp = R[(i-1)*n + _i] * R[(i-1)*n + _j];
                R[_i*n + _j] -= temp;
            }
        }
    }
    // Check for positive definiteness (unless the entire row is zeros, in which case
    // leave the whole row in R as zeros
    if ( R[i*n + i] == 0.0 ) {
        for ( _j = i+1; _j < n; _j ++ ) {
            if (R[i*n + _j] != 0) return 0;
        }
    }
    else if ( R[i*n + i] < -1e-8 ) { // allow for fp error
        return 0;
    }
    else {
        R[i*n + i] = std::sqrt(std::abs( R[i*n + i] ));
        double invR = 1 / R[i*n + i];
        for ( _i = i + 1; _i < n; _i ++ ) {
            R[i*n + _i] *= invR;
        }
    }

    if ( i + 1 < n ) {
        return cholesky_step( R, n, i + 1 );
    }
    return 1;
};

constexpr double L_PLUS_LAMBDA = 3;

namespace utils {
    int cholesky ( int n, double *mat, double *R ) {
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < i; j ++ ) {
                R[n*i + j] = 0;
            }
            for (int j = i; j < n; j ++ ) {
                R[n*i + j] = mat[n*i + j];
            }
        }
        return cholesky_step( R, n, 0 );
    };
    int cholesky ( int n, double scale, double *mat, double *R ) {
        for (int i = 0; i < n; i ++ ) {
            for (int j = 0; j < i; j ++ ) {
                R[n*i + j] = 0;
            }
            for (int j = i; j < n; j ++ ) {
                R[n*i + j] = scale * mat[n*i + j];
            }
        }
        return cholesky_step( R, n, 0 );
    };
    
    int get_sigma_points( int n, double *x, double *cov, double *X ) {
        // Copy in the "mean" vector
        for ( int i = 0; i < n; i ++ ) {
            X[0*n + i] = x[i];
        }

        // Attempt to square-root the scaled covariance
        double R[n*n];
        if ( !cholesky( n, L_PLUS_LAMBDA, cov, R ) ) return 0;
        
        for ( int i = 0; i < n; i ++ ) {
            for ( int j = 0; j < n; j ++ ) {
                X[(i+1)*n   + j] = x[j] + R[i*n + j];
                X[(i+n+1)*n + j] = x[j] - R[i*n + j];
            }
        }

        return 1;
    }

    // n is the length of X[0] and x and cov[0]. L is such that 2*L+1 is length of X
    void GRV_statistics( int n, int L, double *X, double *x, double *cov ) {
        double W0 = (L_PLUS_LAMBDA - L) / L_PLUS_LAMBDA;
        double W = 1 / ( 2 * L_PLUS_LAMBDA );
        int k;

        // Weighted average
        k = 0;
        for ( int j = 0; j < n; j ++ ) {
            x[j] = W0*X[k*n + j];
        } 
        for ( k = 1; k < 2*L+1; k ++ ) {
            for ( int j = 0; j < n; j ++ ) {
                x[j] += W*X[k*n + j];
            }
        } 

        // Covariance
        k = 0;
        for ( int i = 0; i < n; i ++ ) {
            for ( int j = 0; j < n; j ++ ) {
                cov[i*n + j] = W0 * (X[k*n + i] - x[i]) * (X[k*n + j] - x[j]);
            }
        } 
        for ( k = 1; k < 2*L+1; k ++ ) {
            for ( int i = 0; i < n; i ++ ) {
                for ( int j = 0; j < n; j ++ ) {
                    cov[i*n + j] += W * (X[k*n + i] - x[i]) * (X[k*n + j] - x[j]);
                }
            } 
        } 
    }
    void GRV_statistics( int n, double *X, double *x, double *cov ) {
        GRV_statistics(n,n,X,x,cov);
    }
}
