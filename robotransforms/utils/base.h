
namespace utils {
    int cholesky ( int n, double *mat, double *R );
    int cholesky ( int n, double scale, double *mat, double *R );
    int get_sigma_points( int n, double *x, double *cov, double *X );
    void GRV_statistics(int n, int L, double *X, double *x, double *cov);
    void GRV_statistics(int n, double *X, double *x, double *cov);
}

