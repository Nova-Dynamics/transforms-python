#include "base.h"
#include <math.h>
#include <algorithm>

// Many thanks to https://stackoverflow.com/a/9324086/13217806
double clip(double n, double lower, double upper) {
  return std::max(lower, std::min(n, upper));
}

constexpr double PI = 3.14159265358979323846;
constexpr double PI_SQUARED = PI * PI;

namespace euclidean {
    // Conversion
    void convert_quat_to_redquat(double *q, double *rq) {
        double sign = 1;
        if ( q[0] < 0 ) sign = 0;
        rq[0] = sign*q[1];
        rq[1] = sign*q[3];
        rq[2] = sign*q[3];
    }
    void convert_redquat_to_quat(double *rq, double *q) {
        // Always assume positive
        q[0] = std::sqrt(clip(
            1 - rq[0]*rq[0] - rq[1]*rq[1] - rq[2]*rq[2]
        , 0., 1.));
        q[1] = rq[0];
        q[2] = rq[1];
        q[3] = rq[2];
    }
    void convert_quat_to_rotvec(double *q, double *rq) {
        double sign = 1.;
        if ( q[0] < 0 ) sign = -1.;

        double im_norm = std::sqrt(clip(
            q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
        , 0., 1.));

        if ( im_norm == 0. ) {
            rq[0] = sign*0.;
            rq[1] = sign*0.;
            rq[2] = sign*0.;
        } else {
            // If the real part is negative, we need to map q -> -q so that we
            //   can apply our formula (which assumes non-negative real part)
            double asin_times_2_over_norm = 2 * sign * std::asin(im_norm) / im_norm;
            rq[0] = asin_times_2_over_norm * q[1];
            rq[1] = asin_times_2_over_norm * q[2];
            rq[2] = asin_times_2_over_norm * q[3];
        }
    }

    void convert_rotvec_to_quat(double *rq, double *q) {
        // Not a clip just max, becuase the rq could be "added" out of the [-pi, pi] range
        double norm = std::sqrt(std::max(
            rq[0]*rq[0] + rq[1]*rq[1] + rq[2]*rq[2],
            0.
        ));

        if ( norm == 0. ) {
            q[0] = 1.;
            q[1] = 0.;
            q[2] = 0.;
            q[3] = 0.;
        } else {
            double half_norm = norm / 2.0;
            double sin_over_norm = std::sin(half_norm) / norm;
            q[0] = std::cos(half_norm);
            q[1] = rq[0] * sin_over_norm;
            q[2] = rq[1] * sin_over_norm;
            q[3] = rq[2] * sin_over_norm;
        }
    }
    void convert_lrq_to_lrQ(double *lrq, double *lrQ) {
        // The locations are the same
        lrQ[0] = lrq[0];
        lrQ[1] = lrq[1];
        lrQ[2] = lrq[2];
        // upgrade the quaternion part
        convert_redquat_to_quat(&(lrq[3]), &(lrQ[3]));
    }
    void convert_lrQ_to_lrq(double *lrQ, double *lrq) {
        // The locations are the same
        lrq[0] = lrQ[0];
        lrq[1] = lrQ[1];
        lrq[2] = lrQ[2];
        // upgrade the quaternion part
        convert_quat_to_redquat(&(lrQ[3]), &(lrq[3]));
    }
    void convert_lrrv_to_lrQ(double *lrrv, double *lrQ) {
        // The locations are the same
        lrQ[0] = lrrv[0];
        lrQ[1] = lrrv[1];
        lrQ[2] = lrrv[2];
        // upgrade the quaternion part
        convert_rotvec_to_quat(&(lrrv[3]), &(lrQ[3]));
    }
    void convert_lrQ_to_lrrv(double *lrQ, double *lrrv) {
        // The locations are the same
        lrrv[0] = lrQ[0];
        lrrv[1] = lrQ[1];
        lrrv[2] = lrQ[2];
        // upgrade the quaternion part
        convert_quat_to_rotvec(&(lrQ[3]), &(lrrv[3]));
    }

    // Application
    void apply_quat(double *q, double *v, double *w) {
        double vq[4];
        vq[0] =            - v[0]*q[1] - v[1]*q[2] - v[2]*q[3];
        vq[1] =              v[0]*q[0] + v[1]*q[3] - v[2]*q[2];
        vq[2] =              v[1]*q[0] + v[2]*q[1] - v[0]*q[3];
        vq[3] =              v[2]*q[0] + v[0]*q[2] - v[1]*q[1];
        w[0] = q[0]*vq[1] - q[1]*vq[0] - q[2]*vq[3] + q[3]*vq[2];
        w[1] = q[0]*vq[2] - q[2]*vq[0] - q[3]*vq[1] + q[1]*vq[3];
        w[2] = q[0]*vq[3] - q[3]*vq[0] - q[1]*vq[2] + q[2]*vq[1];
    }
    void apply_redquat(double *rq, double *v, double *w) {
        double q[4];
        convert_redquat_to_quat(rq, q);
        apply_quat(q, v, w);
    }
    void apply_rotvec(double *rq, double *v, double *w) {
        double q[4];
        convert_rotvec_to_quat(rq, q);
        apply_quat(q, v, w);
    }
    void apply_lrQ(double *lrQ, double *v, double *w) {
        double t[3];
        t[0] = v[0] + lrQ[0];
        t[1] = v[1] + lrQ[1];
        t[2] = v[2] + lrQ[2];
        apply_quat(&(lrQ[3]), t, w);
    }
    void apply_lrq(double *lrq, double *v, double *w) {
        double t[3];
        t[0] = v[0] + lrq[0];
        t[1] = v[1] + lrq[1];
        t[2] = v[2] + lrq[2];
        apply_redquat(&(lrq[3]), t, w);
    }
    void apply_lrrv(double *lrrv, double *v, double *w) {
        double t[3];
        t[0] = v[0] + lrrv[0];
        t[1] = v[1] + lrrv[1];
        t[2] = v[2] + lrrv[2];
        apply_redquat(&(lrrv[3]), t, w);
    }

    // Inversion
    void invert_quat(double *q1, double *q2) {
        q2[0] =  q1[0];
        q2[1] = -q1[1];
        q2[2] = -q1[2];
        q2[3] = -q1[3];
    }
    void invert_redquat(double *rq1, double *rq2) {
        rq2[0] = -rq1[0];
        rq2[1] = -rq1[1];
        rq2[2] = -rq1[2];
    }
    void invert_rotvec(double *rq1, double *rq2) {
        rq2[0] = -rq1[0];
        rq2[1] = -rq1[1];
        rq2[2] = -rq1[2];
    }
    void invert_lrq(double *lrq1, double *lrq2) {
        invert_redquat(&(lrq1[3]), &(lrq2[3]));
        // Rotate, then flip the sign
        apply_redquat(&(lrq1[3]), lrq1, lrq2);
        lrq2[0] *= -1;
        lrq2[1] *= -1;
        lrq2[2] *= -1;
    }
    void invert_lrrv(double *lrrv1, double *lrrv2) {
        invert_rotvec(&(lrrv1[3]), &(lrrv2[3]));
        // Rotate, then flip the sign
        apply_rotvec(&(lrrv1[3]), lrrv1, lrrv2);
        lrrv2[0] *= -1;
        lrrv2[1] *= -1;
        lrrv2[2] *= -1;
    }
    void invert_lrQ(double *lrQ1, double *lrQ2) {
        invert_quat(&(lrQ1[3]), &(lrQ2[3]));
        // Rotate, then flip the sign
        apply_quat(&(lrQ1[3]), lrQ1, lrQ2);
        lrQ2[0] *= -1;
        lrQ2[1] *= -1;
        lrQ2[2] *= -1;
    }

    // Compositions
    void compose_quat(double *q1, double *q2, double *q3) {
        q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
        q3[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
        q3[2] = q1[0]*q2[2] + q1[2]*q2[0] + q1[3]*q2[1] - q1[1]*q2[3];
        q3[3] = q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1];
    }
    void compose_redquat(double *rq1, double *rq2, double *rq3) {
        double q1[4], q2[4], q3[4];
        convert_redquat_to_quat(rq1, q1);
        convert_redquat_to_quat(rq2, q2);
        compose_quat(q1, q2, q3);
        convert_quat_to_redquat(q3, rq3);
    }
    void compose_rotvec(double *rq1, double *rq2, double *rq3) {
        double q1[4], q2[4], q3[4];
        convert_rotvec_to_quat(rq1, q1);
        convert_rotvec_to_quat(rq2, q2);
        compose_quat(q1, q2, q3);
        convert_quat_to_rotvec(q3, rq3);
    }
    void compose_lrQ(double *lrQ1, double *lrQ2, double *lrQ3) {
        double iQ1[4];
        invert_quat(&(lrQ1[3]), iQ1);
        // Rotate location2 into 1's frame, then add the vectors together
        apply_quat(iQ1, lrQ2, lrQ3);
        lrQ3[0] += lrQ1[0];
        lrQ3[1] += lrQ1[1];
        lrQ3[2] += lrQ1[2];
        // Compose the quaternions
        compose_quat( &(lrQ1[3]), &(lrQ2[3]), &(lrQ3[3]) );
    }
    void compose_lrq(double *lrq1, double *lrq2, double *lrq3) {
        double irq1[3];
        invert_redquat(&(lrq1[3]), irq1);
        // Rotate location2 into 1's frame, then add the vectors together
        apply_redquat(irq1, lrq2, lrq3);
        lrq3[0] += lrq1[0];
        lrq3[1] += lrq1[1];
        lrq3[2] += lrq1[2];
        // Compose the quaternions
        compose_redquat( &(lrq1[3]), &(lrq2[3]), &(lrq3[3]) );
    }
    void compose_lrrv(double *lrrv1, double *lrrv2, double *lrrv3) {
        double irq1[3];
        invert_rotvec(&(lrrv1[3]), irq1);
        // Rotate location2 into 1's frame, then add the vectors together
        apply_rotvec(irq1, lrrv2, lrrv3);
        lrrv3[0] += lrrv1[0];
        lrrv3[1] += lrrv1[1];
        lrrv3[2] += lrrv1[2];
        // Compose the quaternions
        compose_rotvec( &(lrrv1[3]), &(lrrv2[3]), &(lrrv3[3]) );
    }
}
