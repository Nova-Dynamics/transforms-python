#include <math.h>
#include <algorithm>

#include "../utils/base.h"
#include "../euclidean/base.h"
#include "base.h"

constexpr double DR_D_NOM = 0.464;
constexpr double DR_LO2G = 0.05;
constexpr double DR_MU = 0.8;
constexpr double DR_D_EFF_PARAMS[3] = {0.505225138, 4.55440687, 0.00237199585};

// Is this a bad idea?
constexpr double L_PLUS_LAMBDA = 3;

// State vector size
constexpr int SS = 4+3;
constexpr int ESS = 4+3+2;
constexpr int SM = 3+3;
constexpr int ESM = 3+3+2;

double sinc(double x) {
    // Use power series for small x
    if (std::abs(x) < 1e-4) return 1 - (x*x/6) + (x*x*x*x/120);
    return std::sin(x)/x;
}

namespace dead_reckon {
    double dr_calculate_d(double v_ave, double v_diff) {
        double _v_diff = std::max(1e-8, v_diff);
        double v_ratio = v_ave / (_v_diff * DR_D_EFF_PARAMS[1]);
        return DR_D_NOM
            + (DR_D_EFF_PARAMS[0] - DR_D_NOM) * std::exp( -0.5 * v_ratio * v_ratio )
            + DR_D_EFF_PARAMS[2] * std::abs(v_ave);
    }

    void dead_reckon_step( double *quat, double dl, double dr, double vave, double vdiff, double *delta ) {
        double lstar = 0.5 * ( dl + dr );
        double to2 = ( dl - dr ) / ( 2 * dr_calculate_d(vave, vdiff) );
        double chord = lstar * sinc(to2);
        double sin = std::sin(to2);
        double cos = std::cos(to2);
        double xr = 0;
        if ( std::abs(lstar) > 1e-3 && std::abs(vave) > 1e-6 ) {
            double rho = 2 * to2 / lstar;
            xr = DR_LO2G * vave * vave * rho * rho / DR_MU;
        }

        double x[3] = {
            chord * ( sin + cos*xr ),
            chord * ( cos - sin*xr ),
            0
        };

        double iq[4];
        euclidean::invert_quat(quat, iq);
        euclidean::apply_quat(iq, x, delta);
    }

    void dead_reckon_step_errors( double dl, double dr, double vave, double vdiff, double dl_scale, double *out ) {

        double ddl = (dl_scale * std::abs(dl));
        double ddr = (dl_scale * std::abs(dr));
        double rho = 1 - 1e-8;
        if ( std::abs(vdiff) >= 1e-3 ) {
            rho = std::abs(std::tanh( vave / vdiff ));
        }

        out[0] = ddl*ddl + 1e-8;
        out[1] = rho*ddl*ddr;
        out[2] = ddr*ddr + 1e-8;
    }

    int dead_reckon_apply(double* step, double *x, double *cov) {
        if ( step[1] == 0 && step[2] == 0 ) return 1;
        // Create extended state vector as state + dl + dr
        double z[ESS] = {
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            x[6],
            step[1], // dl
            step[2], // dr
        };

        double dr_cov[3];
        dead_reckon_step_errors(
                step[1], step[2], // dl, dr
                step[7], step[8], // vave, vdiff
                0.005,
                dr_cov);          // output

        // Create extended covariance matrix
        double P_z[ESM*ESM] = {
            cov[0*SM + 0],cov[0*SM + 1],cov[0*SM + 2],cov[0*SM + 3],cov[0*SM + 4],cov[0*SM + 5],0,0,
            cov[1*SM + 0],cov[1*SM + 1],cov[1*SM + 2],cov[1*SM + 3],cov[1*SM + 4],cov[1*SM + 5],0,0,
            cov[2*SM + 0],cov[2*SM + 1],cov[2*SM + 2],cov[2*SM + 3],cov[2*SM + 4],cov[2*SM + 5],0,0,
            cov[3*SM + 0],cov[3*SM + 1],cov[3*SM + 2],cov[3*SM + 3],cov[3*SM + 4],cov[3*SM + 5],0,0,
            cov[4*SM + 0],cov[4*SM + 1],cov[4*SM + 2],cov[4*SM + 3],cov[4*SM + 4],cov[4*SM + 5],0,0,
            cov[5*SM + 0],cov[5*SM + 1],cov[5*SM + 2],cov[5*SM + 3],cov[5*SM + 4],cov[5*SM + 5],0,0,
            0,0,0,0,0,0,dr_cov[0],dr_cov[1],
            0,0,0,0,0,0,dr_cov[1],dr_cov[2],
        };

        // Get the sigma points
        // n = length of single sigma point: ESS
        // L = length size of covariance: ESM
        int n = ESS;
        int L = ESM;
        double Z[(2*L+1)*n];
        for ( int i = 0; i < n; i ++ ) {
            Z[0*n + i] = z[i]; // copy in mean
        }
        // Attempt to square-root the scaled covariance
        double R[L*L];
        if ( !utils::cholesky( L, L_PLUS_LAMBDA, P_z, R ) ) return 0;
        
        for ( int i = 0; i < L; i ++ ) {
            // Extract and lrQ from the ith row of the root covariance (which is a manifold deviation)
            double dlrQ[7], idlrQ[7];
            euclidean::convert_lrq_to_lrQ(&R[i*L], dlrQ);
            euclidean::invert_lrQ(dlrQ, idlrQ);

            // Write into the the sigma point the difference
            euclidean::compose_lrQ(z, dlrQ,  &Z[(i+1)*n]);
            euclidean::compose_lrQ(z, idlrQ, &Z[(i+L+1)*n]);


            // Also add in the last two elements
            Z[(i+1)*n + SS + 0] = z[SS + 0] + R[i*L + SM + 0];
            Z[(i+1)*n + SS + 1] = z[SS + 1] + R[i*L + SM + 1];
            Z[(i+L+1)*n + SS + 0] = z[SS + 0] - R[i*L + SM + 0];
            Z[(i+L+1)*n + SS + 1] = z[SS + 1] - R[i*L + SM + 1];
        }

        // Transform the sigma points
        double Y[(2*L+1)*3];
        for ( int i = 0; i < 2*L+1; i ++ ) {
            dead_reckon_step(
                &Z[i*n + 3], //quat
                z[SS + 0],   // dl
                z[SS + 1],   // dr
                step[7],     // vave
                step[8],     // vave
                &Y[i*3]
            );
            for ( int j = 0; j < 3; j ++ ) {
                Y[i*3 + j] += Z[i*n + j];
            }
        }

        // Note, this probably only works becuase we have neglected the quat -- not sure how to average on the manifold....
        double y[3], y_cov[3*3];
        utils::GRV_statistics(3, L, Y, y, y_cov);


        // Update state and covarience
        x[0] = Y[0]; // use mean transform
        x[1] = Y[1]; // use mean transform
        x[2] = Y[2]; // use mean transform
        double quat[4];
        euclidean::compose_quat(&x[3], &step[3], quat); // accumulate difference along manifold
        x[3] = quat[0];
        x[4] = quat[1];
        x[5] = quat[2];
        x[6] = quat[3];

        for ( int i = 0; i < 3; i ++ ) {
            for ( int j = 0; j < 3; j ++ ) {
                cov[i*SM + j] = y_cov[i*3 + j];
            }
        }
        for ( int i = 0; i < 3; i ++ ) {
            for ( int j = 3; j < SM; j ++ ) {
                cov[i*SM + j] = 0;
                cov[j*SM + i] = 0;
            }
        }
        for ( int i = 3; i < SM; i ++ ) {
            cov[i*SM + i] = 1e-5; // TODO : this is a rough estimate of IMU's accuracy
        }

        return 1;
    }
    int dead_reckon_apply(double* step, double *x, double *cov, double *x_new, double *cov_new ) {
        // Copy data in
        for ( int i = 0; i < SS; i ++ ) {
            x_new[i] = x[i];
        }
        for ( int i = 0; i < SM; i ++ ) {
            for ( int j = 0; j < SM; j ++ ) {
                cov_new[i*SM + j] = cov[i*SM + j];
            }
        }
        return dead_reckon_apply( step, x_new, cov_new );
    }

    int dead_reckon(int n_steps, double* steps, double *x, double *P) {
        for ( int i = 0; i < n_steps; i ++ ) {
            if ( !dead_reckon_apply( &steps[i*9], x, P ) ) return 0;
        }
        return 1;
    }
    int dead_reckon(int n_steps, double* steps, double *x, double *P, double *x_new, double *P_new  ) {
        // Copy data in
        for ( int i = 0; i < SS; i ++ ) {
            x_new[i] = x[i];
        }
        for ( int i = 0; i < SM; i ++ ) {
            for ( int j = 0; j < SM; j ++ ) {
                P_new[i*SM + j] = P[i*SM + j];
            }
        }
        return dead_reckon( n_steps, steps, x_new, P_new );
    }
}
