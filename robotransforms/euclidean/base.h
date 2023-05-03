
namespace euclidean {
    // Conversion
    void convert_quat_to_redquat(double *q, double *rq);
    void convert_redquat_to_quat(double *rq, double *q);
    void convert_quat_to_rotvec(double *q, double *rq);
    void convert_rotvec_to_quat(double *rq, double *q);
    void convert_lrq_to_lrQ(double *lrq, double *lrQ);
    void convert_lrQ_to_lrq(double *lrQ, double *lrq);
    void convert_lrrv_to_lrQ(double *lrrv, double *lrQ);
    void convert_lrQ_to_lrrv(double *lrQ, double *lrrv);
    // Application
    void apply_quat(double *q, double *v, double *w);
    void apply_redquat(double *rq, double *v, double *w);
    void apply_rotvec(double *rq, double *v, double *w);
    void apply_lrQ(double *lrQ, double *v, double *w);
    void apply_lrq(double *lrq, double *v, double *w);
    void apply_lrrv(double *lrrv, double *v, double *w);
    // Inversion
    void invert_quat(double *q1, double *q2);
    void invert_redquat(double *rq1, double *rq2);
    void invert_rotvec(double *rq1, double *rq2);
    void invert_lrQ(double *lrQ1, double *lrQ2);
    void invert_lrq(double *lrq1, double *lrq2);
    void invert_lrrv(double *lrrv1, double *lrrv2);
    // Compositions
    void compose_quat(double *q1, double *q2, double *q3);
    void compose_redquat(double *rq1, double *rq2, double *rq3);
    void compose_rotvec(double *rq1, double *rq2, double *rq3);
    void compose_lrQ(double *lrQ1, double *lrQ2, double *lrq3);
    void compose_lrq(double *lrq1, double *lrq2, double *lrq3);
    void compose_lrrv(double *lrrv1, double *lrrv2, double *lrrv3);
}
