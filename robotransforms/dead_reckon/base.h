
namespace dead_reckon {
    double dr_calculate_d(double v_ave, double v_diff);
    // quat is 4, delta is 3
    void dead_reckon_step(double* quat, double dl, double dr, double vave, double vdiff, double *delta );
    void dead_reckon_step_errors( double dl, double dr, double vave, double vdiff, double dl_scale, double* out);
    // Step is length 9, x is length 7 and cov is length 6
    int dead_reckon_apply(double* step, double *x, double *cov );
    int dead_reckon_apply(double* step, double *x, double *cov, double *x_new, double *cov_new );
    int dead_reckon(int n_steps, double* steps, double *x, double *P);
    int dead_reckon(int n_steps, double* steps, double *x, double *P, double *x_new, double *cov_new  );
}

