
namespace euclidean {
    // Conversion
    void convert_quat_to_redquat(double *q, double *rq);
    void convert_redquat_to_quat(double *rq, double *q);
    void convert_lrq_to_lrQ(double *lrq, double *lrQ);
    void convert_lrQ_to_lrq(double *lrQ, double *lrq);
    // Application
    void apply_quat(double *q, double *v, double *w);
    void apply_redquat(double *rq, double *v, double *w);
    void apply_lrQ(double *lrQ, double *v, double *w);
    void apply_lrq(double *lrq, double *v, double *w);
    // Inversion
    void invert_quat(double *q1, double *q2);
    void invert_redquat(double *rq1, double *rq2);
    void invert_lrQ(double *lrQ1, double *lrQ2);
    void invert_lrq(double *lrq1, double *lrq2);
    // Compositions
    void compose_quat(double *q1, double *q2, double *q3);
    void compose_redquat(double *rq1, double *rq2, double *rq3);
    void compose_lrQ(double *lrQ1, double *lrQ2, double *lrq3);
    void compose_lrq(double *lrq1, double *lrq2, double *lrq3);
}
